import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# -------------------------- 导入你的生成函数/类/配置（核心） --------------------------
from Car_generate import (
    generate_netcar_locations,
    NetCarLocation,
    CAR_NUM,
    COORDINATE_PRECISION  # 按需导入（非必需，仅做兼容）
)
from Order_generate import (
    generate_taxi_orders,
    TaxiOrder,
    ORDER_NUM,
    cal_km_by_lon_lat  # 复用你的距离计算函数
)


class TaxiCarClusterManager:
    """
    出租车订单与网约车位置的聚类匹配管理器
    核心功能：
    1. 订单/车辆KMeans聚类
    2. 聚类中心匈牙利算法匹配
    3. 子群体拆分
    4. 聚类结果可视化（并排展示）
    5. 结果打印与JSON持久化存储
    """

    def __init__(
        self, 
        n_clusters: int = 4,
        storage_dir: str = "./cluster_subgroups",
        json_file_name: str = "all_subgroups.json",
        random_state: int = 42
    ):
        """
        初始化聚类管理器
        :param n_clusters: 聚类数量（订单/车辆统一使用）
        :param storage_dir: JSON结果存储目录
        :param json_file_name: 汇总JSON文件名
        :param random_state: 随机种子（保证KMeans可复现）
        """
        # 配置参数
        self.n_clusters = n_clusters
        self.storage_dir = storage_dir
        self.json_file_name = json_file_name
        self.random_state = random_state

        # 中间结果存储
        self.clustered_orders: List[TaxiOrder] = []  # 聚类后的订单
        self.clustered_cars: List[NetCarLocation] = []  # 聚类后的车辆
        self.order_kmeans: Optional[KMeans] = None  # 订单聚类模型
        self.car_kmeans: Optional[KMeans] = None  # 车辆聚类模型
        self.order_features: Optional[np.ndarray] = None  # 订单特征矩阵（中点坐标）
        self.car_features: Optional[np.ndarray] = None  # 车辆特征矩阵（实时坐标）
        self.center_matches: Dict[int, int] = {}  # 订单聚类中心→车辆聚类中心 匹配关系
        self.match_distances: Dict[int, float] = {}  # 聚类中心匹配距离
        self.subgroups: Dict[int, Tuple[List[TaxiOrder], List[NetCarLocation]]] = {}  # 拆分后的子群体

    # -------------------------- 基础工具方法 --------------------------
    @staticmethod
    def calculate_order_midpoint(start_lon: float, start_lat: float, end_lon: float, end_lat: float) -> Tuple[float, float]:
        """计算订单起点-终点的中点坐标（聚类专用）"""
        mid_lon = (start_lon + end_lon) / 2
        mid_lat = (start_lat + end_lat) / 2
        return mid_lon, mid_lat

    @staticmethod
    def calculate_euclidean_distance(coord1: np.ndarray, coord2: np.ndarray) -> float:
        """计算两点间的欧氏距离（聚类专用）"""
        return np.linalg.norm(coord1 - coord2)

    # -------------------------- 聚类核心方法 --------------------------
    def cluster_taxi_orders(self, orders: List[TaxiOrder]) -> List[TaxiOrder]:
        """
        订单聚类（基于中点坐标，动态添加聚类属性）
        :param orders: 原始订单列表
        :return: 聚类后的订单列表
        """
        if not orders:
            raise ValueError("订单列表不能为空！")

        # 提取订单中点特征并动态添加属性
        features = []
        for order in orders:
            mid_lon, mid_lat = self.calculate_order_midpoint(
                order.start_lon, order.start_lat,
                order.end_lon, order.end_lat
            )
            order.mid_lon = mid_lon
            order.mid_lat = mid_lat
            features.append([mid_lon, mid_lat])
        
        self.order_features = np.array(features)

        # 执行KMeans聚类
        self.order_kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        cluster_labels = self.order_kmeans.fit_predict(self.order_features)

        # 为订单添加聚类标签
        for idx, order in enumerate(orders):
            order.cluster_label = cluster_labels[idx]

        self.clustered_orders = orders
        return self.clustered_orders

    def cluster_netcar_locations(self, locations: List[NetCarLocation]) -> List[NetCarLocation]:
        """
        网约车聚类（基于实时坐标，动态添加聚类属性）
        :param locations: 原始车辆位置列表
        :return: 聚类后的车辆列表
        """
        if not locations:
            raise ValueError("车辆位置列表不能为空！")

        # 提取车辆坐标特征
        features = []
        for loc in locations:
            features.append([loc.lon, loc.lat])
        
        self.car_features = np.array(features)

        # 执行KMeans聚类
        self.car_kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        cluster_labels = self.car_kmeans.fit_predict(self.car_features)

        # 为车辆添加聚类标签
        for idx, loc in enumerate(locations):
            loc.cluster_label = cluster_labels[idx]

        self.clustered_cars = locations
        return self.clustered_cars

    # -------------------------- 聚类中心匹配与子群体拆分 --------------------------
    def match_cluster_centers(self) -> Tuple[Dict[int, int], Dict[int, float]]:
        """
        基于匈牙利算法匹配订单和汽车聚类中心（一一对应）
        注意：需先执行cluster_taxi_orders和cluster_netcar_locations
        :return: (中心匹配关系, 匹配距离)
        """
        if self.order_kmeans is None or self.car_kmeans is None:
            raise RuntimeError("请先执行订单和车辆的聚类操作！")

        # 计算聚类中心距离矩阵
        order_centers = self.order_kmeans.cluster_centers_
        car_centers = self.car_kmeans.cluster_centers_
        distance_matrix = cdist(order_centers, car_centers, metric='euclidean')

        # 匈牙利算法最优匹配
        order_indices, car_indices = linear_sum_assignment(distance_matrix)

        # 构建匹配结果
        center_matches = {}
        match_distances = {}
        for o_idx, c_idx in zip(order_indices, car_indices):
            center_matches[o_idx] = c_idx
            match_distances[o_idx] = distance_matrix[o_idx, c_idx]
        
        self.center_matches = center_matches
        self.match_distances = match_distances
        return self.center_matches, self.match_distances

    def split_into_subgroups(self) -> Dict[int, Tuple[List[TaxiOrder], List[NetCarLocation]]]:
        """
        根据聚类中心匹配结果拆分子群体
        注意：需先执行match_cluster_centers
        :return: 子群体字典 {子群体ID: (订单列表, 车辆列表)}
        """
        if not self.center_matches:
            raise RuntimeError("请先执行聚类中心匹配操作！")
        if not self.clustered_orders or not self.clustered_cars:
            raise RuntimeError("请先完成订单和车辆的聚类！")

        subgroups = {}
        for subgroup_id, (order_cluster_id, car_cluster_id) in enumerate(self.center_matches.items()):
            subgroup_orders = [o for o in self.clustered_orders if o.cluster_label == order_cluster_id]
            subgroup_cars = [c for c in self.clustered_cars if c.cluster_label == car_cluster_id]
            subgroups[subgroup_id] = (subgroup_orders, subgroup_cars)
        
        self.subgroups = subgroups
        return self.subgroups

    # -------------------------- 可视化方法 --------------------------
    def plot_clustering_side_by_side(self, figsize: Tuple[int, int] = (16, 8)):
        """
        并排绘制订单和网约车的聚类结果（自动使用实例中的聚类数据）
        :param figsize: 画布尺寸
        """
        if self.order_features is None or self.car_features is None:
            raise RuntimeError("请先执行订单和车辆的聚类操作！")
        if not self.clustered_orders or not self.clustered_cars:
            raise RuntimeError("聚类结果为空，无法可视化！")

        # 提取聚类标签和中心
        order_labels = [o.cluster_label for o in self.clustered_orders]
        car_labels = [c.cluster_label for c in self.clustered_cars]
        order_centers = self.order_kmeans.cluster_centers_ if self.order_kmeans else None
        car_centers = self.car_kmeans.cluster_centers_ if self.car_kmeans else None

        # 创建并排子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 绘制订单聚类子图
        scatter1 = ax1.scatter(
            self.order_features[:, 0], self.order_features[:, 1],
            c=order_labels, cmap='viridis', alpha=0.6
        )
        ax1.set_title(f"Order Clustering (K={self.n_clusters})", fontsize=14)
        ax1.set_xlabel("lon", fontsize=12)
        ax1.set_ylabel("lat", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.5)
        if order_centers is not None:
            ax1.scatter(
                order_centers[:, 0], order_centers[:, 1],
                c='red', s=200, marker='*', label='Cluster Center'
            )
            ax1.legend(fontsize=10)
        cbar1 = fig.colorbar(scatter1, ax=ax1, label='Order Cluster')
        cbar1.ax.tick_params(labelsize=10)

        # 绘制网约车聚类子图
        scatter2 = ax2.scatter(
            self.car_features[:, 0], self.car_features[:, 1],
            c=car_labels, cmap='viridis', alpha=0.6
        )
        ax2.set_title(f"NetCar Clustering (K={self.n_clusters})", fontsize=14)
        ax2.set_xlabel("lon", fontsize=12)
        ax2.set_ylabel("lat", fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.5)
        if car_centers is not None:
            ax2.scatter(
                car_centers[:, 0], car_centers[:, 1],
                c='red', s=200, marker='*', label='Cluster Center'
            )
            ax2.legend(fontsize=10)
        cbar2 = fig.colorbar(scatter2, ax=ax2, label='Car Cluster')
        cbar2.ax.tick_params(labelsize=10)

        plt.tight_layout()
        plt.show()

    # -------------------------- 结果打印方法 --------------------------
    def print_matching_results(self):
        """打印聚类中心匹配结果"""
        if not self.center_matches or not self.match_distances:
            raise RuntimeError("请先执行聚类中心匹配操作！")

        print("=" * 50)
        print("聚类中心匹配结果（订单中心 → 汽车中心）")
        print("=" * 50)
        for order_id, car_id in self.center_matches.items():
            distance = self.match_distances[order_id]
            print(f"订单聚类中心{order_id} → 汽车聚类中心{car_id} | 距离: {distance:.6f}")

    def print_subgroup_stats(self):
        """打印子群体拆分结果统计"""
        if not self.subgroups:
            raise RuntimeError("请先执行子群体拆分操作！")

        print("\n" + "=" * 50)
        print("子群体拆分结果统计")
        print("=" * 50)
        total_orders = 0
        total_cars = 0
        for subgroup_id, (orders, cars) in self.subgroups.items():
            order_count = len(orders)
            car_count = len(cars)
            total_orders += order_count
            total_cars += car_count
            print(f"子群体{subgroup_id}: 订单数={order_count} | 汽车数={car_count}")
        print(f"总计: 订单数={total_orders} | 汽车数={total_cars}")



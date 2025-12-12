import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# 导入外部文件的类和函数（请确保Car_generate.py/Order_generate.py存在）
from Car_generate import (
    generate_netcar_locations,
    NetCarLocation,
    CAR_NUM  # 导入配置常量
)
from Order_generate import (
    generate_taxi_orders,
    TaxiOrder,
    ORDER_NUM  # 导入配置常量
)

# -------------------------- 核心配置 --------------------------
N_CLUSTERS = 6  # 统一订单和汽车的聚类数量
STORAGE_DIR = "./cluster_subgroups"  # JSON数据存储目录
JSON_FILE_NAME = "all_subgroups.json"  # 汇总JSON文件名

# -------------------------- 基础工具函数 --------------------------
def calculate_order_midpoint(start_lon: float, start_lat: float, end_lon: float, end_lat: float) -> Tuple[float, float]:
    """计算订单起点-终点的中点坐标"""
    mid_lon = (start_lon + end_lon) / 2
    mid_lat = (start_lat + end_lat) / 2
    return mid_lon, mid_lat

def calculate_euclidean_distance(coord1: np.ndarray, coord2: np.ndarray) -> float:
    """计算两点间的欧氏距离"""
    return np.linalg.norm(coord1 - coord2)

# -------------------------- 可视化函数 --------------------------
def plot_clustering_side_by_side(
    order_features: np.ndarray, order_labels: list, order_title: str,
    car_features: np.ndarray, car_labels: list, car_title: str,
    order_centers: np.ndarray = None, car_centers: np.ndarray = None
):
    """并排绘制订单和网约车的聚类结果，标注聚类中心"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 绘制订单聚类子图
    scatter1 = ax1.scatter(order_features[:, 0], order_features[:, 1], c=order_labels, cmap='viridis', alpha=0.6)
    ax1.set_title(order_title, fontsize=14)
    ax1.set_xlabel("lon", fontsize=12)
    ax1.set_ylabel("lat", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    if order_centers is not None:
        ax1.scatter(order_centers[:, 0], order_centers[:, 1], c='red', s=200, marker='*', label='Cluster Center')
        ax1.legend(fontsize=10)
    cbar1 = fig.colorbar(scatter1, ax=ax1, label='Order Cluster')
    cbar1.ax.tick_params(labelsize=10)

    # 绘制网约车聚类子图
    scatter2 = ax2.scatter(car_features[:, 0], car_features[:, 1], c=car_labels, cmap='viridis', alpha=0.6)
    ax2.set_title(car_title, fontsize=14)
    ax2.set_xlabel("lon", fontsize=12)
    ax2.set_ylabel("lat", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    if car_centers is not None:
        ax2.scatter(car_centers[:, 0], car_centers[:, 1], c='red', s=200, marker='*', label='Cluster Center')
        ax2.legend(fontsize=10)
    cbar2 = fig.colorbar(scatter2, ax=ax2, label='Car Cluster')
    cbar2.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.show()

# -------------------------- 聚类核心逻辑 --------------------------
def cluster_taxi_orders(orders: List[TaxiOrder], n_clusters: int) -> Tuple[List[TaxiOrder], KMeans, np.ndarray]:
    """订单聚类（基于中点坐标）"""
    features = []
    for order in orders:
        mid_lon, mid_lat = calculate_order_midpoint(
            order.start_lon, order.start_lat,
            order.end_lon, order.end_lat
        )
        order.mid_lon, order.mid_lat = mid_lon, mid_lat
        features.append([mid_lon, mid_lat])
    features = np.array(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    for idx, order in enumerate(orders):
        order.cluster_label = cluster_labels[idx]

    return orders, kmeans, features

def cluster_netcar_locations(locations: List[NetCarLocation], n_clusters: int) -> Tuple[List[NetCarLocation], KMeans, np.ndarray]:
    """网约车聚类（基于实时坐标）"""
    features = []
    for loc in locations:
        features.append([loc.lon, loc.lat])
    features = np.array(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    for idx, loc in enumerate(locations):
        loc.cluster_label = cluster_labels[idx]

    return locations, kmeans, features

# -------------------------- 聚类中心匹配与子群体拆分 --------------------------
def match_cluster_centers(order_centers: np.ndarray, car_centers: np.ndarray) -> Tuple[Dict[int, int], Dict[int, float]]:
    """基于匈牙利算法匹配订单和汽车聚类中心（一一对应）"""
    distance_matrix = cdist(order_centers, car_centers, metric='euclidean')
    order_indices, car_indices = linear_sum_assignment(distance_matrix)

    center_matches = {}
    match_distances = {}
    for o_idx, c_idx in zip(order_indices, car_indices):
        center_matches[o_idx] = c_idx
        match_distances[o_idx] = distance_matrix[o_idx, c_idx]
    
    return center_matches, match_distances

def split_into_subgroups(
    clustered_orders: List[TaxiOrder],
    clustered_cars: List[NetCarLocation],
    center_matches: Dict[int, int]
) -> Dict[int, Tuple[List[TaxiOrder], List[NetCarLocation]]]:
    """根据聚类中心匹配结果拆分子群体"""
    subgroups = {}
    for subgroup_id, (order_cluster_id, car_cluster_id) in enumerate(center_matches.items()):
        subgroup_orders = [o for o in clustered_orders if o.cluster_label == order_cluster_id]
        subgroup_cars = [c for c in clustered_cars if c.cluster_label == car_cluster_id]
        subgroups[subgroup_id] = (subgroup_orders, subgroup_cars)
    return subgroups

# -------------------------- 结果打印函数 --------------------------
def print_matching_results(center_matches: Dict[int, int], match_distances: Dict[int, float]):
    """打印聚类中心匹配结果"""
    print("=" * 50)
    print("聚类中心匹配结果（订单中心 → 汽车中心）")
    print("=" * 50)
    for order_id, car_id in center_matches.items():
        distance = match_distances[order_id]
        print(f"订单聚类中心{order_id} → 汽车聚类中心{car_id} | 距离: {distance:.6f}")

def print_subgroup_stats(subgroups: Dict[int, Tuple[List[TaxiOrder], List[NetCarLocation]]]):
    """打印子群体统计信息"""
    print("\n" + "=" * 50)
    print("子群体拆分结果统计")
    print("=" * 50)
    total_orders = 0
    total_cars = 0
    for subgroup_id, (orders, cars) in subgroups.items():
        order_count = len(orders)
        car_count = len(cars)
        total_orders += order_count
        total_cars += car_count
        print(f"子群体{subgroup_id}: 订单数={order_count} | 汽车数={car_count}")
    print(f"总计: 订单数={total_orders} | 汽车数={total_cars}")

# -------------------------- JSON存储核心函数 --------------------------
def init_storage_dir():
    """初始化存储目录（不存在则创建）"""
    Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    return STORAGE_DIR

def taxi_order_to_dict(order: TaxiOrder) -> Dict:
    """将TaxiOrder对象转为可序列化的字典"""
    return {
        "order_id": getattr(order, "order_id", None),  # 适配你的Order类实际属性
        "start_lon": float(order.start_lon),
        "start_lat": float(order.start_lat),
        "end_lon": float(order.end_lon),
        "end_lat": float(order.end_lat),
        "mid_lon": float(order.mid_lon) if hasattr(order, "mid_lon") else None,
        "mid_lat": float(order.mid_lat) if hasattr(order, "mid_lat") else None,
        "cluster_label": int(order.cluster_label) if hasattr(order, "cluster_label") else None
    }

def netcar_location_to_dict(car: NetCarLocation) -> Dict:
    """将NetCarLocation对象转为可序列化的字典"""
    return {
        "car_id": getattr(car, "car_id", None),  # 适配你的Car类实际属性
        "lon": float(car.lon),
        "lat": float(car.lat),
        "cluster_label": int(car.cluster_label) if hasattr(car, "cluster_label") else None
    }

def save_subgroup_to_json(subgroups: Dict[int, Tuple[List[TaxiOrder], List[NetCarLocation]]]):
    """将所有子群体数据保存为JSON文件"""
    storage_dir = init_storage_dir()
    json_path = f"{storage_dir}/{JSON_FILE_NAME}"
    
    # 构建JSON数据结构
    json_data = {
        "cluster_config": {"n_clusters": N_CLUSTERS},
        "subgroups": {}
    }
    
    # 填充子群体数据
    for subgroup_id, (orders, cars) in subgroups.items():
        json_data["subgroups"][str(subgroup_id)] = {
            "order_count": len(orders),
            "car_count": len(cars),
            "orders": [taxi_order_to_dict(o) for o in orders],
            "cars": [netcar_location_to_dict(c) for c in cars]
        }
    
    # 写入JSON文件（格式化输出，便于阅读）
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nJSON数据已保存至：{json_path}")
    return json_path

# -------------------------- 主执行逻辑 --------------------------
if __name__ == "__main__":
    # 1. 生成订单并聚类
    taxi_orders = generate_taxi_orders(ORDER_NUM)
    clustered_orders, order_kmeans, order_features = cluster_taxi_orders(taxi_orders, N_CLUSTERS)
    order_labels = [o.cluster_label for o in clustered_orders]
    order_centers = order_kmeans.cluster_centers_
    order_title = f"Order Clustering (K={N_CLUSTERS})"

    # 2. 生成网约车并聚类
    car_locations = generate_netcar_locations(CAR_NUM)
    clustered_cars, car_kmeans, car_features = cluster_netcar_locations(car_locations, N_CLUSTERS)
    car_labels = [c.cluster_label for c in clustered_cars]
    car_centers = car_kmeans.cluster_centers_
    car_title = f"NetCar Clustering (K={N_CLUSTERS})"

    # 3. 聚类中心匹配
    center_matches, match_distances = match_cluster_centers(order_centers, car_centers)
    print_matching_results(center_matches, match_distances)

    # 4. 拆分子群体
    subgroups = split_into_subgroups(clustered_orders, clustered_cars, center_matches)
    print_subgroup_stats(subgroups)

    # 5. 可视化聚类结果
    plot_clustering_side_by_side(
        order_features, order_labels, order_title,
        car_features, car_labels, car_title,
        order_centers=order_centers,
        car_centers=car_centers
    )

    # 6. 保存子群体数据为JSON（核心输出）
    save_subgroup_to_json(subgroups)

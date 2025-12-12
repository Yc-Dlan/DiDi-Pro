import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import List, Tuple  # 新增：导入Tuple类型（兼容全版本）

# 导入外部文件的类和函数（核心解耦逻辑）
from Car_generate import (
    generate_netcar_locations,
    save_netcar_locations,
    NetCarLocation,
    CAR_NUM  # 导入配置常量
)
from Order_generate import (
    generate_taxi_orders,
    save_taxi_orders,
    TaxiOrder,
    ORDER_NUM  # 导入配置常量
)

# -------------------------- 聚类配置 --------------------------
N_CLUSTERS_ORDERS = 6    # 订单聚类数量
N_CLUSTERS_CARS = 6      # 网约车聚类数量

# -------------------------- 聚类工具函数 --------------------------
def calculate_order_midpoint(start_lon: float, start_lat: float, end_lon: float, end_lat: float) -> Tuple[float, float]:
    """计算订单起点-终点的中点坐标"""
    mid_lon = (start_lon + end_lon) / 2
    mid_lat = (start_lat + end_lat) / 2
    return mid_lon, mid_lat

def plot_clustering_side_by_side(
    order_features: np.ndarray, order_labels: list, order_title: str,
    car_features: np.ndarray, car_labels: list, car_title: str
):
    """并排绘制订单和网约车的聚类结果（1行2列）"""
    # 创建画布（宽16，高8，适配并排布局）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 绘制订单聚类子图
    scatter1 = ax1.scatter(order_features[:, 0], order_features[:, 1], c=order_labels, cmap='viridis', alpha=0.6)
    ax1.set_title(order_title, fontsize=14)
    ax1.set_xlabel("lon", fontsize=12)
    ax1.set_ylabel("lat", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    # 为订单子图添加颜色条（单独的颜色条，避免共享）
    cbar1 = fig.colorbar(scatter1, ax=ax1, label='1')
    cbar1.ax.tick_params(labelsize=10)

    # 绘制网约车聚类子图
    scatter2 = ax2.scatter(car_features[:, 0], car_features[:, 1], c=car_labels, cmap='viridis', alpha=0.6)
    ax2.set_title(car_title, fontsize=14)
    ax2.set_xlabel("lon", fontsize=12)
    ax2.set_ylabel("lat", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    # 为网约车子图添加颜色条
    cbar2 = fig.colorbar(scatter2, ax=ax2, label='2')
    cbar2.ax.tick_params(labelsize=10)

    # 调整子图间距，避免标题/标签重叠
    plt.tight_layout()
    # 显示图像
    plt.show()

# -------------------------- 核心聚类逻辑 --------------------------
def cluster_taxi_orders(orders: List[TaxiOrder], n_clusters: int) -> Tuple[List[TaxiOrder], KMeans, np.ndarray]:
    """订单聚类（基于中点坐标）"""
    # 提取中点特征
    features = []
    for order in orders:
        mid_lon, mid_lat = calculate_order_midpoint(
            order.start_lon, order.start_lat,
            order.end_lon, order.end_lat
        )
        order.mid_lon, order.mid_lat = mid_lon, mid_lat
        features.append([mid_lon, mid_lat])
    features = np.array(features)

    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    # 赋值聚类标签
    for idx, order in enumerate(orders):
        order.cluster_label = cluster_labels[idx]

    return orders, kmeans, features

def cluster_netcar_locations(locations: List[NetCarLocation], n_clusters: int) -> Tuple[List[NetCarLocation], KMeans, np.ndarray]:
    """网约车聚类（基于实时坐标）"""
    # 提取坐标特征
    features = []
    for loc in locations:
        features.append([loc.lon, loc.lat])
    features = np.array(features)

    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    # 赋值聚类标签
    for idx, loc in enumerate(locations):
        loc.cluster_label = cluster_labels[idx]

    return locations, kmeans, features

# -------------------------- 主执行逻辑 --------------------------
if __name__ == "__main__":

    # 生成订单（调用外部函数）
    taxi_orders = generate_taxi_orders(ORDER_NUM)
    # 聚类
    clustered_orders, _, order_features = cluster_taxi_orders(taxi_orders, N_CLUSTERS_ORDERS)

    # 提取订单聚类标签
    order_labels = [o.cluster_label for o in clustered_orders]
    order_title = f"order({N_CLUSTERS_ORDERS})"

    # 生成网约车位置（调用外部函数）
    car_locations = generate_netcar_locations(CAR_NUM)
    # 聚类
    clustered_cars, _, car_features = cluster_netcar_locations(car_locations, N_CLUSTERS_CARS)

    # 提取网约车聚类标签
    car_labels = [c.cluster_label for c in clustered_cars]
    car_title = f"car({N_CLUSTERS_CARS})"

    # ========== 3. 并排绘制两张聚类图 ==========
    plot_clustering_side_by_side(
        order_features, order_labels, order_title,
        car_features, car_labels, car_title
    )

import numpy as np
import matplotlib.pyplot as plt
from Car_generate import generate_netcar_locations, CAR_NUM
from Order_generate import generate_taxi_orders, ORDER_NUM
from K_means_visual import TaxiCarClusterManager  # 导入聚类管理器类

def visualize_clustering_and_matching(n_clusters=4):
    """
    完整流程：生成数据 → 执行聚类 → 匹配中心 → 可视化聚类分布和中心匹配关系
    """
    # 1. 生成模拟数据
    orders = generate_taxi_orders(ORDER_NUM)  # 生成订单数据
    cars = generate_netcar_locations(CAR_NUM)  # 生成车辆数据
    print(f"已生成 {len(orders)} 个订单和 {len(cars)} 辆车辆数据")

    # 2. 初始化聚类管理器并执行聚类
    cluster_manager = TaxiCarClusterManager(n_clusters=n_clusters, random_state=42)
    cluster_manager.cluster_taxi_orders(orders)  # 订单聚类（基于起点-终点中点）
    cluster_manager.cluster_netcar_locations(cars)  # 车辆聚类（基于实时位置）
    center_matches, match_distances = cluster_manager.match_cluster_centers()  # 匹配聚类中心

    # 3. 可视化1：订单与车辆的聚类分布（并排展示）
    cluster_manager.plot_clustering_side_by_side(figsize=(16, 8))

    # 4. 可视化2：聚类中心匹配关系（连线展示）
    plt.figure(figsize=(10, 8))
    
    # 获取聚类中心坐标
    order_centers = cluster_manager.order_kmeans.cluster_centers_
    car_centers = cluster_manager.car_kmeans.cluster_centers_
    
    # 绘制订单聚类中心（蓝色星号）
    plt.scatter(
        order_centers[:, 0], order_centers[:, 1],
        c='blue', s=300, marker='*', label='Order Cluster Centers'
    )
    # 绘制车辆聚类中心（红色方块）
    plt.scatter(
        car_centers[:, 0], car_centers[:, 1],
        c='red', s=300, marker='s', label='Car Cluster Centers'
    )
    
    # 绘制匹配连线（灰色虚线）
    for order_cluster_id, car_cluster_id in center_matches.items():
        o_center = order_centers[order_cluster_id]
        c_center = car_centers[car_cluster_id]
        plt.plot(
            [o_center[0], c_center[0]],  # 经度连线
            [o_center[1], c_center[1]],  # 纬度连线
            'gray', linestyle='--', linewidth=2,
            label=f"Match {order_cluster_id}→{car_cluster_id}" if order_cluster_id == 0 else ""
        )
    
    # 图表配置
    plt.title(f"Cluster Center Matches (K={n_clusters})", fontsize=14)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    # 5. 打印匹配结果统计
    cluster_manager.print_matching_results()
    cluster_manager.split_into_subgroups()
    cluster_manager.print_subgroup_stats()

if __name__ == "__main__":
    # 可自定义聚类数量（如4、5、6等）
    visualize_clustering_and_matching(n_clusters=4)

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from Distance_transfer import cal_km_by_lon_lat

from Car_generate import (
    generate_netcar_locations,
    NetCarLocation,
    CAR_NUM,
    COORDINATE_PRECISION 
)
from Order_generate import (
    generate_taxi_orders,
    TaxiOrder,
    ORDER_NUM,
)

class TaxiCarClusterMatcher:
    """出租车订单与网约车位置的聚类匹配器"""
    def __init__(self, n_clusters = 4, random_state = 42):
        """
        :param n_clusters: 聚类数量
        :param random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.random_state = random_state

        # 存储中间结果
        self.clustered_orders = []  # 聚类后的订单列表
        self.clustered_cars = []  # 聚类后的车辆列表
        self.order_kmeans = None  # 订单聚类模型
        self.car_kmeans = None  # 车辆聚类模型
        self.order_features = None  # 订单聚类特征矩阵（中点坐标）
        self.car_features = None  # 车辆聚类特征矩阵（实时坐标）
        self.center_matches = {}  # 订单聚类中心 -> 车辆聚类中心 匹配关系
        self.match_distances = {}  # 聚类中心匹配的距离
        self.subgroups = {}  # 拆分后的子群体

    def calculate_order_midpoint(self, start_lon, start_lat, end_lon, end_lat):
        """计算订单起点-终点的中点坐标"""
        mid_lon = (start_lon + end_lon) / 2
        mid_lat = (start_lat + end_lat) / 2
        return mid_lon, mid_lat

    def calculate_euclidean_distance(self, coord1, coord2):
        """计算两点间的欧氏距离"""
        return np.linalg.norm(coord1 - coord2)

    def cluster_taxi_orders(self, orders):
        """
        订单聚类
        :param orders: 原始订单列表
        :return: 聚类后的订单列表
        """
        # 提取订单中点特征
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

        # KMeans聚类
        self.order_kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        cluster_labels = self.order_kmeans.fit_predict(self.order_features)

        # 为订单添加聚类标签
        for idx, order in enumerate(orders):
            order.cluster_label = cluster_labels[idx]

        self.clustered_orders = orders
        return self.clustered_orders

    def cluster_netcar_locations(self, locations):
        """
        网约车聚类
        :param locations: 原始车辆位置列表
        :return: 聚类后的车辆列表
        """
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

    def match_cluster_centers(self):
        """
        匈牙利算法匹配
        return: (中心匹配关系, 匹配距离)
        """
        # 获取聚类中心并计算距离矩阵
        order_centers = self.order_kmeans.cluster_centers_
        car_centers = self.car_kmeans.cluster_centers_
        distance_matrix = cdist(order_centers, car_centers, metric='euclidean')

        # 匈牙利算法求解最优匹配
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

    def split_into_subgroups(self):
        """
        根据聚类中心匹配结果拆分子群体
        return: 子群体字典
        """
        subgroups = {}
        for subgroup_id, (order_cluster_id, car_cluster_id) in enumerate(self.center_matches.items()):
            # 筛选对应聚类标签的订单和车辆
            subgroup_orders = [o for o in self.clustered_orders if o.cluster_label == order_cluster_id]
            subgroup_cars = [c for c in self.clustered_cars if c.cluster_label == car_cluster_id]
            subgroups[subgroup_id] = (subgroup_orders, subgroup_cars)
        
        self.subgroups = subgroups
        return self.subgroups


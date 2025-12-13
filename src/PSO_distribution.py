import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# -------------------------- 导入外部解耦的函数/类 --------------------------
# 假设你的聚类核心逻辑在 cluster_analysis.py 中，按需调整文件名
from KMEA import (
    # 数据生成函数
    generate_taxi_orders,
    generate_netcar_locations,
    # 数据类
    TaxiOrder,
    NetCarLocation,
    # 聚类核心函数
    cluster_taxi_orders,
    cluster_netcar_locations,
    match_cluster_centers,
    split_into_subgroups,
    # 配置常量（也可在本脚本重新定义）
    N_CLUSTERS,
    CAR_NUM,
    ORDER_NUM
)

# -------------------------- PSO专属配置（仅在本脚本定义） --------------------------
PSO_CONFIG = {
    "n_particles": 50,       # 粒子数量
    "max_iter": 100,         # 最大迭代次数
    "w": 0.7,                # 惯性权重
    "c1": 1.5,               # 个体学习因子
    "c2": 1.5,               # 全局学习因子
    "v_max": 5.0,            # 最大速度
}
STORAGE_DIR = "./pso_matching_results"  # PSO结果独立存储目录

# -------------------------- PSO核心工具函数 --------------------------
def calculate_haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """哈维正弦公式计算经纬度距离（千米），适配地理坐标"""
    R = 6371  # 地球半径（千米）
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# -------------------------- 子群内PSO匹配核心类 --------------------------
class SubgroupPSOMatcher:
    """子群内订单-车辆匹配的PSO算法实现（无耦合，仅依赖输入的订单/车辆列表）"""
    def __init__(self, subgroup_id: int, orders: List[TaxiOrder], cars: List[NetCarLocation]):
        self.subgroup_id = subgroup_id
        self.orders = orders
        self.cars = cars
        self.n_orders = len(orders)
        self.n_cars = len(cars)
        self.n_match = min(self.n_orders, self.n_cars)  # 匹配数取最小值

        # 预处理：车辆到订单起点的距离矩阵
        self.distance_matrix = self._build_distance_matrix()

        # PSO参数（从配置读取）
        self.n_particles = PSO_CONFIG["n_particles"]
        self.max_iter = PSO_CONFIG["max_iter"]
        self.w = PSO_CONFIG["w"]
        self.c1 = PSO_CONFIG["c1"]
        self.c2 = PSO_CONFIG["c2"]
        self.v_max = PSO_CONFIG["v_max"]

        # PSO最优结果缓存
        self.best_global_fitness = float('inf')
        self.best_global_position = None

    def _build_distance_matrix(self) -> np.ndarray:
        """构建车辆-订单起点距离矩阵（仅依赖输入的订单/车辆坐标）"""
        dist_mat = np.zeros((self.n_cars, self.n_orders))
        for car_idx, car in enumerate(self.cars):
            for order_idx, order in enumerate(self.orders):
                dist_mat[car_idx, order_idx] = calculate_haversine_distance(
                    car.lon, car.lat, order.start_lon, order.start_lat
                )
        return dist_mat

    def _init_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """初始化粒子群（位置=订单索引，速度=随机值）"""
        positions = np.zeros((self.n_particles, self.n_match), dtype=int)
        velocities = np.random.uniform(-self.v_max, self.v_max, (self.n_particles, self.n_match))

        for i in range(self.n_particles):
            if self.n_orders >= self.n_match:
                # 订单充足：随机不重复选订单
                positions[i] = np.random.choice(self.n_orders, size=self.n_match, replace=False)
            else:
                # 订单不足：部分车辆无订单（-1标记）
                pos = np.random.choice(self.n_orders, size=self.n_orders, replace=False)
                padding = -1 * np.ones(self.n_match - self.n_orders, dtype=int)
                positions[i] = np.concatenate([pos, padding])
        return positions, velocities
    
    def _calculate_fitness(self, position: np.ndarray) -> float:
        """优化后：匹配数量权重70%，距离权重30%，平衡匹配数和距离"""
        # 1. 计算匹配数量（越多越好，取倒数转为「越小越好」）
        matched_count = sum(1 for o in position if o != -1)
        count_score = 1 / (matched_count + 1e-6)  # 避免除0
    
        # 2. 计算总距离（越小越好，归一化）
        total_dist = 0.0
        for car_idx, order_idx in enumerate(position):
            if order_idx == -1:
                continue
            total_dist += self.distance_matrix[car_idx, order_idx]
        dist_score = total_dist / (self.n_match * np.max(self.distance_matrix) + 1e-6)  # 归一化到0-1
    
        # 3. 加权求和（可调整权重）
        return 0.3 * dist_score + 0.7 * count_score  # 优先保证匹配数量


    def _update_velocity(self, vel: np.ndarray, pos: np.ndarray, p_best: np.ndarray, g_best: np.ndarray) -> np.ndarray:
        """更新粒子速度（标准PSO公式）"""
        r1, r2 = np.random.random(vel.shape), np.random.random(vel.shape)
        new_vel = self.w * vel + self.c1 * r1 * (p_best - pos) + self.c2 * r2 * (g_best - pos)
        return np.clip(new_vel, -self.v_max, self.v_max)

    def _update_position(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """更新粒子位置（保证合法且不重复）"""
        new_pos = pos + vel.astype(int)
        # 边界限制
        new_pos = np.where(new_pos < -1, -1, new_pos)
        new_pos = np.where(new_pos >= self.n_orders, self.n_orders - 1, new_pos)

        # 去重：一个订单仅匹配一辆车
        for i in range(len(new_pos)):
            if new_pos[i] == -1:
                continue
            # 检查当前订单是否已被前面的车辆占用
            if new_pos[i] in new_pos[:i]:
                # 选未被占用的订单，无则标记为-1
                available = [o for o in range(self.n_orders) if o not in new_pos[:i]]
                new_pos[i] = available[0] if available else -1
        return new_pos

    def run(self) -> Tuple[Dict[int, Optional[TaxiOrder]], float]:
        """执行PSO并返回匹配结果（无耦合，仅输出字典）"""
        if self.n_match == 0:
            return {}, 0.0

        # 初始化粒子
        positions, velocities = self._init_particles()
        p_best_pos = positions.copy()
        p_best_fit = np.array([self._calculate_fitness(p) for p in positions])

        # 全局最优初始化
        g_best_idx = np.argmin(p_best_fit)
        self.best_global_position = p_best_pos[g_best_idx].copy()
        self.best_global_fitness = p_best_fit[g_best_idx]

        # 迭代优化
        for iter_idx in range(self.max_iter):
            for i in range(self.n_particles):
                current_fit = self._calculate_fitness(positions[i])

                # 更新个体最优
                if current_fit < p_best_fit[i]:
                    p_best_fit[i] = current_fit
                    p_best_pos[i] = positions[i].copy()

                # 更新全局最优
                if current_fit < self.best_global_fitness:
                    self.best_global_fitness = current_fit
                    self.best_global_position = positions[i].copy()

                # 更新速度和位置
                velocities[i] = self._update_velocity(velocities[i], positions[i], p_best_pos[i], self.best_global_position)
                positions[i] = self._update_position(positions[i], velocities[i])

            # 打印进度
            if (iter_idx + 1) % 20 == 0:
                print(f"子群{self.subgroup_id} PSO迭代{iter_idx+1}/{self.max_iter} | 最优距离：{self.best_global_fitness:.2f}km")

        # 生成匹配结果（仅返回索引和对象映射，无外部依赖）
        match_result = {}
        for car_idx in range(self.n_cars):
            if car_idx < len(self.best_global_position):
                order_idx = self.best_global_position[car_idx]
                match_result[car_idx] = self.orders[order_idx] if (order_idx != -1 and order_idx < self.n_orders) else None
            else:
                match_result[car_idx] = None

        return match_result, self.best_global_fitness

# -------------------------- PSO结果保存函数（独立） --------------------------
def save_pso_results(subgroups, pso_results):
    """保存PSO匹配结果（仅依赖基础数据类型，无耦合）"""
    Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    save_path = f"{STORAGE_DIR}/pso_matching_results.json"

    # 构建纯字典格式的结果（避免对象序列化问题）
    json_data = {
        "config": {
            "n_clusters": N_CLUSTERS,
            "pso_config": PSO_CONFIG,
            "total_orders": ORDER_NUM,
            "total_cars": CAR_NUM
        },
        "subgroups": {}
    }

    for subgroup_id, (orders, cars) in subgroups.items():
        match_result, total_dist = pso_results[subgroup_id]
        subgroup_data = {
            "subgroup_id": subgroup_id,
            "order_count": len(orders),
            "car_count": len(cars),
            "matched_count": sum(1 for v in match_result.values() if v is not None),
            "total_travel_distance_km": round(total_dist, 2),
            "matches": {}
        }

        # 转换为纯基础类型（避免对象序列化）
        for car_idx, order in match_result.items():
            if order is not None:
                subgroup_data["matches"][str(car_idx)] = {
                    "car_lon": cars[car_idx].lon,
                    "car_lat": cars[car_idx].lat,
                    "order_start_lon": order.start_lon,
                    "order_start_lat": order.start_lat,
                    "order_end_lon": order.end_lon,
                    "order_end_lat": order.end_lat,
                    "travel_distance_km": round(calculate_haversine_distance(
                        cars[car_idx].lon, cars[car_idx].lat,
                        order.start_lon, order.start_lat
                    ), 2)
                }
            else:
                subgroup_data["matches"][str(car_idx)] = None

        json_data["subgroups"][str(subgroup_id)] = subgroup_data

    # 保存JSON
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    print(f"\nPSO匹配结果已保存至：{save_path}")

# -------------------------- 主执行逻辑（仅PSO相关） --------------------------
def main():
    """主函数：仅执行子群内PSO订单匹配（无聚类逻辑耦合，聚类通过导入函数完成）"""
    # 1. 导入聚类后的子群体（聚类逻辑完全解耦，仅调用函数）
    print("===== 1. 生成数据并获取子群体（聚类逻辑解耦） =====")
    # 生成原始数据
    taxi_orders = generate_taxi_orders(ORDER_NUM)
    car_locations = generate_netcar_locations(CAR_NUM)
    # 执行聚类（调用外部函数，本脚本不包含聚类逻辑）
    clustered_orders, order_kmeans, _ = cluster_taxi_orders(taxi_orders, N_CLUSTERS)
    clustered_cars, car_kmeans, _ = cluster_netcar_locations(car_locations, N_CLUSTERS)
    # 匹配聚类中心并拆分子群
    center_matches, _ = match_cluster_centers(order_kmeans.cluster_centers_, car_kmeans.cluster_centers_)
    subgroups = split_into_subgroups(clustered_orders, clustered_cars, center_matches)

    # 2. 子群内PSO匹配（核心逻辑，无任何耦合）
    print("\n===== 2. 执行子群内PSO订单匹配 =====")
    pso_results = {}
    for subgroup_id, (orders, cars) in subgroups.items():
        print(f"\n处理子群{subgroup_id}：订单数={len(orders)}，车辆数={len(cars)}")
        if len(orders) == 0 or len(cars) == 0:
            pso_results[subgroup_id] = ({}, 0.0)
            print(f"子群{subgroup_id}无订单/车辆，跳过")
            continue

        # 仅执行PSO匹配（核心逻辑）
        matcher = SubgroupPSOMatcher(subgroup_id, orders, cars)
        match_result, total_dist = matcher.run()
        pso_results[subgroup_id] = (match_result, total_dist)

        # 打印子群匹配结果
        matched_count = sum(1 for v in match_result.values() if v is not None)
        print(f"子群{subgroup_id}匹配完成：匹配{matched_count}单，总距离{total_dist:.2f}km")

    # 3. 保存结果（仅保存基础数据，无对象依赖）
    save_pso_results(subgroups, pso_results)

    # 4. 汇总结果
    print("\n===== 3. PSO匹配结果汇总 =====")
    total_matched = sum(sum(1 for v in res[0].values() if v is not None) for res in pso_results.values())
    total_distance = sum(res[1] for res in pso_results.values())
    print(f"全局总匹配订单数：{total_matched}/{ORDER_NUM}")
    print(f"全局总行驶距离：{total_distance:.2f}km")
    if total_matched > 0:
        print(f"平均每单行驶距离：{total_distance/total_matched:.2f}km")

if __name__ == "__main__":
    main()

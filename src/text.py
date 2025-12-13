import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from itertools import permutations

# -------------------------- 导入外部解耦的函数/类 --------------------------
from KMEA import (
    generate_taxi_orders,
    generate_netcar_locations,
    TaxiOrder,
    NetCarLocation,
    cluster_taxi_orders,
    cluster_netcar_locations,
    match_cluster_centers,
    split_into_subgroups,
    N_CLUSTERS,
    CAR_NUM,
    ORDER_NUM
)

# -------------------------- 全局配置 --------------------------
# PSO专属配置
PSO_CONFIG = {
    "n_particles": 50,       # 粒子数量
    "max_iter": 100,         # 最大迭代次数
    "w": 0.7,                # 惯性权重
    "c1": 1.5,               # 个体学习因子
    "c2": 1.5,               # 全局学习因子
    "v_max": 5.0,            # 最大速度
}
# 拼车配置
MAX_PASSENGERS = 4          # 每车最大乘客数
MATCH_ROUNDS = 3            # 最大接单轮次（支持二次/三次接单）
STORAGE_DIR = "./pso_carpool_matching_results"  # 结果存储目录

# 可视化配置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 负号显示
sns.set_style("whitegrid")
COLORS = sns.color_palette("husl", n_colors=N_CLUSTERS)  # 子群配色
CAR_COLORS = sns.color_palette("tab10", n_colors=CAR_NUM)  # 车辆配色

# -------------------------- 核心工具函数 --------------------------
def calculate_haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """哈维正弦公式计算经纬度距离（千米）"""
    R = 6371  # 地球半径（千米）
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def optimize_carpool_route(car_lon: float, car_lat: float, orders: List[TaxiOrder]) -> Tuple[List[TaxiOrder], float]:
    """优化拼车接驾路径：计算最优接送顺序，返回排序后的订单和总距离"""
    if len(orders) <= 1:
        total_dist = calculate_haversine_distance(car_lon, car_lat, orders[0].start_lon, orders[0].start_lat) if orders else 0.0
        return orders, total_dist
    
    # 枚举所有可能的接驾顺序，选择总距离最短的
    min_dist = float('inf')
    best_order = []
    for perm in permutations(orders):
        current_lon, current_lat = car_lon, car_lat
        total_dist = 0.0
        for order in perm:
            # 从当前位置到订单起点
            total_dist += calculate_haversine_distance(current_lon, current_lat, order.start_lon, order.start_lat)
            current_lon, current_lat = order.start_lon, order.start_lat
        if total_dist < min_dist:
            min_dist = total_dist
            best_order = list(perm)
    return best_order, min_dist

# -------------------------- 拼车+多轮接单PSO匹配类 --------------------------
class CarpoolPSOMatcher:
    """支持拼车（≤4人）和多轮接单的PSO匹配器"""
    def __init__(self, subgroup_id: int, orders: List[TaxiOrder], cars: List[NetCarLocation]):
        self.subgroup_id = subgroup_id
        self.initial_orders = orders.copy()  # 初始订单列表
        self.cars = cars.copy()              # 车辆列表
        self.n_cars = len(cars)
        
        # 订单状态追踪（确保全部分配）
        self.unassigned_orders: Set[int] = set(range(len(orders)))
        # 最终匹配结果：{车辆索引: [(轮次1订单列表, 起点, 终点), (轮次2订单列表, 起点, 终点), ...]}
        self.final_matches: Dict[int, List[Tuple[List[TaxiOrder], Tuple[float, float], Tuple[float, float]]]] = {
            car_idx: [] for car_idx in range(self.n_cars)
        }
        
        # PSO基础参数
        self.n_particles = PSO_CONFIG["n_particles"]
        self.max_iter = PSO_CONFIG["max_iter"]
        self.w = PSO_CONFIG["w"]
        self.c1 = PSO_CONFIG["c1"]
        self.c2 = PSO_CONFIG["c2"]
        self.v_max = PSO_CONFIG["v_max"]

    def _build_distance_matrix(self, available_orders: List[int]) -> np.ndarray:
        """构建车辆到可用订单起点的距离矩阵"""
        dist_mat = np.zeros((self.n_cars, len(available_orders)))
        for car_idx, car in enumerate(self.cars):
            for ord_idx, global_ord_idx in enumerate(available_orders):
                order = self.initial_orders[global_ord_idx]
                dist_mat[car_idx, ord_idx] = calculate_haversine_distance(
                    car.lon, car.lat, order.start_lon, order.start_lat
                )
        return dist_mat

    def _init_particles(self, n_available: int, n_cars: int) -> Tuple[np.ndarray, np.ndarray]:
        """初始化粒子群：粒子位置=每车分配的订单数（≤4），速度=随机值"""
        # 粒子维度：(n_cars, MAX_PASSENGERS)，值=订单索引（-1表示无订单）
        positions = -1 * np.ones((self.n_particles, n_cars, MAX_PASSENGERS), dtype=int)
        velocities = np.random.uniform(-self.v_max, self.v_max, (self.n_particles, n_cars, MAX_PASSENGERS))

        for i in range(self.n_particles):
            # 随机分配可用订单，确保不重复、每车≤4单
            assigned = set()
            for car_idx in range(n_cars):
                if not n_available:
                    break
                # 每车最多分配min(剩余订单, 4)单
                n_assign = min(len(self.unassigned_orders) - len(assigned), MAX_PASSENGERS)
                if n_assign <= 0:
                    continue
                # 随机选订单
                available = list(self.unassigned_orders - assigned)
                if not available:
                    break
                assign_orders = np.random.choice(available, size=n_assign, replace=False)
                positions[i, car_idx, :n_assign] = assign_orders
                assigned.update(assign_orders)
        return positions, velocities
    
    def _calculate_fitness(self, position: np.ndarray, available_orders: List[int], dist_mat: np.ndarray) -> float:
        """优化目标：100%订单完成率优先 + 拼车距离 + 车辆利用率"""
        # 1. 订单完成率（必须100%，权重60%）
        assigned_orders = set()
        for car_idx in range(position.shape[0]):
            for ord_idx in position[car_idx]:
                if ord_idx != -1 and ord_idx in self.unassigned_orders:
                    assigned_orders.add(ord_idx)
        completion_rate = len(assigned_orders) / len(available_orders) if available_orders else 1.0
        completion_score = 1 - completion_rate  # 越小越好
        
        # 2. 总行驶距离（权重30%）
        total_dist = 0.0
        for car_idx in range(position.shape[0]):
            car_orders = [o for o in position[car_idx] if o != -1 and o in self.unassigned_orders]
            if not car_orders:
                continue
            # 计算拼车最优路径距离
            car = self.cars[car_idx]
            orders_obj = [self.initial_orders[o] for o in car_orders]
            _, route_dist = optimize_carpool_route(car.lon, car.lat, orders_obj)
            total_dist += route_dist
        # 归一化距离得分
        dist_score = total_dist / (len(available_orders) * np.max(dist_mat) + 1e-6) if available_orders else 0.0
        
        # 3. 车辆利用率（权重10%）：尽量少用车，多拼车
        used_cars = sum(1 for car_idx in range(position.shape[0]) if any(o != -1 for o in position[car_idx]))
        utilization_score = used_cars / self.n_cars if self.n_cars else 0.0
        
        # 加权求和（完成率优先）
        return 0.6 * completion_score + 0.3 * dist_score + 0.1 * utilization_score

    def _update_velocity(self, vel: np.ndarray, pos: np.ndarray, p_best: np.ndarray, g_best: np.ndarray) -> np.ndarray:
        """更新粒子速度"""
        r1, r2 = np.random.random(vel.shape), np.random.random(vel.shape)
        new_vel = self.w * vel + self.c1 * r1 * (p_best - pos) + self.c2 * r2 * (g_best - pos)
        return np.clip(new_vel, -self.v_max, self.v_max)

    def _update_position(self, pos: np.ndarray, available_orders: List[int]) -> np.ndarray:
        """更新粒子位置：保证订单不重复、每车≤4单"""
        new_pos = pos.copy().astype(int)
        global_assigned = set()
        
        for car_idx in range(new_pos.shape[0]):
            # 过滤无效订单
            car_orders = [o for o in new_pos[car_idx] if o in available_orders and o not in global_assigned]
            # 去重并限制数量≤4
            car_orders = list(dict.fromkeys(car_orders))[:MAX_PASSENGERS]
            # 补全-1
            padding = -1 * np.ones(MAX_PASSENGERS - len(car_orders), dtype=int)
            new_pos[car_idx] = np.concatenate([car_orders, padding])
            # 更新全局已分配
            global_assigned.update(car_orders)
        
        return new_pos

    def _single_round_match(self, start_positions: Dict[int, Tuple[float, float]]) -> Dict[int, List[TaxiOrder]]:
        """单轮匹配：基于当前车辆位置匹配订单，返回每车匹配的订单"""
        if not self.unassigned_orders:
            return {car_idx: [] for car_idx in range(self.n_cars)}
        
        available_orders = list(self.unassigned_orders)
        n_available = len(available_orders)
        dist_mat = self._build_distance_matrix(available_orders)
        
        # 初始化粒子
        positions, velocities = self._init_particles(n_available, self.n_cars)
        p_best_pos = positions.copy()
        p_best_fit = np.array([self._calculate_fitness(p, available_orders, dist_mat) for p in positions])
        
        # 全局最优初始化
        g_best_idx = np.argmin(p_best_fit)
        best_global_pos = p_best_pos[g_best_idx].copy()
        best_global_fit = p_best_fit[g_best_idx]

        # 迭代优化
        for iter_idx in range(self.max_iter):
            for i in range(self.n_particles):
                current_fit = self._calculate_fitness(positions[i], available_orders, dist_mat)
                
                # 更新个体最优
                if current_fit < p_best_fit[i]:
                    p_best_fit[i] = current_fit
                    p_best_pos[i] = positions[i].copy()
                
                # 更新全局最优
                if current_fit < best_global_fit:
                    best_global_fit = current_fit
                    best_global_pos = positions[i].copy()
                
                # 更新速度和位置
                velocities[i] = self._update_velocity(velocities[i], positions[i], p_best_pos[i], best_global_pos)
                positions[i] = self._update_position(positions[i], available_orders)
            
            if (iter_idx + 1) % 20 == 0:
                print(f"子群{self.subgroup_id} 迭代{iter_idx+1}/{self.max_iter} | 最优适应度：{best_global_fit:.4f}")

        # 生成本轮匹配结果
        round_matches = {}
        for car_idx in range(self.n_cars):
            car_orders_idx = [o for o in best_global_pos[car_idx] if o != -1 and o in self.unassigned_orders]
            car_orders = [self.initial_orders[o] for o in car_orders_idx]
            
            # 优化拼车路径
            car_lon, car_lat = start_positions[car_idx]
            optimized_orders, _ = optimize_carpool_route(car_lon, car_lat, car_orders)
            
            round_matches[car_idx] = optimized_orders
            # 标记已分配订单
            self.unassigned_orders -= set([self.initial_orders.index(o) for o in optimized_orders])
        
        return round_matches

    def run_multi_round_matching(self) -> Dict[int, List[Tuple[List[TaxiOrder], Tuple[float, float], Tuple[float, float]]]]:
        """执行多轮匹配：支持二次/三次接单，直到所有订单完成"""
        print(f"\n开始子群{self.subgroup_id}多轮拼车匹配（总订单数：{len(self.initial_orders)}）")
        
        # 初始化车辆位置（初始位置）
        car_current_pos = {car_idx: (car.lon, car.lat) for car_idx, car in enumerate(self.cars)}
        
        # 多轮匹配
        for round_idx in range(MATCH_ROUNDS):
            if not self.unassigned_orders:
                print(f"子群{self.subgroup_id} 第{round_idx+1}轮：所有订单已分配，提前结束")
                break
            
            print(f"\n子群{self.subgroup_id} 第{round_idx+1}轮匹配（剩余订单：{len(self.unassigned_orders)}）")
            # 单轮匹配
            round_matches = self._single_round_match(car_current_pos)
            
            # 更新最终匹配结果和车辆位置
            for car_idx, orders in round_matches.items():
                if not orders:
                    continue
                # 记录本轮匹配（订单列表、起点、终点）
                start_pos = car_current_pos[car_idx]
                # 终点=最后一个订单的终点
                end_pos = (orders[-1].end_lon, orders[-1].end_lat)
                self.final_matches[car_idx].append((orders, start_pos, end_pos))
                # 更新车辆位置到本轮终点
                car_current_pos[car_idx] = end_pos
            
            print(f"子群{self.subgroup_id} 第{round_idx+1}轮完成：分配{len(self.initial_orders)-len(self.unassigned_orders)}/{len(self.initial_orders)}单")
        
        # 兜底：确保所有订单都被分配（处理极端情况）
        if self.unassigned_orders:
            print(f"子群{self.subgroup_id} 兜底分配剩余{len(self.unassigned_orders)}单")
            remaining_orders = [self.initial_orders[o] for o in self.unassigned_orders]
            car_idx = 0
            while remaining_orders and car_idx < self.n_cars:
                # 每车补单至4人
                current_orders = self.final_matches[car_idx][-1][0] if self.final_matches[car_idx] else []
                n_need = min(MAX_PASSENGERS - len(current_orders), len(remaining_orders))
                add_orders = remaining_orders[:n_need]
                if add_orders:
                    if self.final_matches[car_idx]:
                        # 追加到最后一轮
                        last_round = self.final_matches[car_idx][-1]
                        new_orders = last_round[0] + add_orders
                        self.final_matches[car_idx][-1] = (new_orders, last_round[1], (new_orders[-1].end_lon, new_orders[-1].end_lat))
                    else:
                        # 新增轮次
                        start_pos = car_current_pos[car_idx]
                        end_pos = (add_orders[-1].end_lon, add_orders[-1].end_lat)
                        self.final_matches[car_idx].append((add_orders, start_pos, end_pos))
                    remaining_orders = remaining_orders[n_need:]
                car_idx += 1
        
        print(f"子群{self.subgroup_id} 匹配完成：总订单{len(self.initial_orders)}，已分配{len(self.initial_orders)-len(self.unassigned_orders)}单")
        return self.final_matches

# -------------------------- 结果保存函数 --------------------------
def save_carpool_results(subgroups: Dict[int, Tuple[List[TaxiOrder], List[NetCarLocation]]],
                         carpool_results: Dict[int, Dict[int, List[Tuple[List[TaxiOrder], Tuple[float, float], Tuple[float, float]]]]]):
    """保存拼车匹配结果为JSON"""
    Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    save_path = f"{STORAGE_DIR}/carpool_matching_results.json"

    json_data = {
        "config": {
            "n_clusters": N_CLUSTERS,
            "pso_config": PSO_CONFIG,
            "max_passengers": MAX_PASSENGERS,
            "match_rounds": MATCH_ROUNDS,
            "total_orders": ORDER_NUM,
            "total_cars": CAR_NUM
        },
        "subgroups": {}
    }

    for subgroup_id, (orders, cars) in subgroups.items():
        subgroup_matches = carpool_results[subgroup_id]
        subgroup_data = {
            "subgroup_id": subgroup_id,
            "total_orders": len(orders),
            "total_cars": len(cars),
            "car_matches": {}
        }

        for car_idx, round_matches in subgroup_matches.items():
            car_data = {
                "car_lon": cars[car_idx].lon if car_idx < len(cars) else 0.0,
                "car_lat": cars[car_idx].lat if car_idx < len(cars) else 0.0,
                "rounds": []
            }
            
            for round_idx, (orders_list, start_pos, end_pos) in enumerate(round_matches):
                round_data = {
                    "round_idx": round_idx + 1,
                    "start_lon": start_pos[0],
                    "start_lat": start_pos[1],
                    "end_lon": end_pos[0],
                    "end_lat": end_pos[1],
                    "passenger_count": len(orders_list),
                    "orders": [],
                    "total_travel_distance": 0.0
                }
                
                # 计算本轮总距离
                current_lon, current_lat = start_pos
                total_dist = 0.0
                for order in orders_list:
                    # 接驾距离
                    dist = calculate_haversine_distance(current_lon, current_lat, order.start_lon, order.start_lat)
                    total_dist += dist
                    current_lon, current_lat = order.start_lon, order.start_lat
                    # 送驾距离（到订单终点）
                    dist = calculate_haversine_distance(current_lon, current_lat, order.end_lon, order.end_lat)
                    total_dist += dist
                    current_lon, current_lat = order.end_lon, order.end_lat
                round_data["total_travel_distance"] = round(total_dist, 2)
                
                # 订单详情
                for order in orders_list:
                    round_data["orders"].append({
                        "order_start_lon": order.start_lon,
                        "order_start_lat": order.start_lat,
                        "order_end_lon": order.end_lon,
                        "order_end_lat": order.end_lat,
                        "pickup_distance": round(calculate_haversine_distance(start_pos[0], start_pos[1], order.start_lon, order.start_lat), 2)
                    })
                
                car_data["rounds"].append(round_data)
            
            subgroup_data["car_matches"][str(car_idx)] = car_data

        json_data["subgroups"][str(subgroup_id)] = subgroup_data

    # 保存JSON
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    print(f"\n拼车匹配结果已保存至：{save_path}")

# -------------------------- 可视化函数 --------------------------
def plot_carpool_cluster_distribution(subgroups: Dict[int, Tuple[List[TaxiOrder], List[NetCarLocation]]], 
                                      carpool_results: Dict[int, Dict[int, List[Tuple[List[TaxiOrder], Tuple[float, float], Tuple[float, float]]]]],
                                      save_dir: str):
    """绘制拼车聚类分布+接单路径"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for subgroup_id, (orders, cars) in subgroups.items():
        color = COLORS[subgroup_id % len(COLORS)]
        subgroup_matches = carpool_results[subgroup_id]
        
        # 绘制订单（起点=蓝色，终点=红色）
        order_start_lons = [o.start_lon for o in orders]
        order_start_lats = [o.start_lat for o in orders]
        order_end_lons = [o.end_lon for o in orders]
        order_end_lats = [o.end_lat for o in orders]
        ax.scatter(order_start_lons, order_start_lats, color=color, s=60, alpha=0.6, 
                   label=f'子群{subgroup_id}订单起点', marker='o', edgecolors='black', linewidth=0.5)
        ax.scatter(order_end_lons, order_end_lats, color=color, s=60, alpha=0.6, 
                   label=f'子群{subgroup_id}订单终点', marker='x', edgecolors='black', linewidth=0.5)
        
        # 绘制车辆和接单路径
        for car_idx, round_matches in subgroup_matches.items():
            car_color = CAR_COLORS[car_idx % len(CAR_COLORS)]
            # 绘制车辆初始位置
            car = cars[car_idx]
            ax.scatter(car.lon, car.lat, color=car_color, s=100, alpha=0.9, 
                       label=f'车辆{car_idx}初始位置' if subgroup_id == 0 and car_idx == 0 else "", 
                       marker='^', edgecolors='black', linewidth=1)
            
            # 绘制接单路径
            for round_idx, (orders_list, start_pos, end_pos) in enumerate(round_matches):
                # 起点到第一个订单
                if orders_list:
                    first_order = orders_list[0]
                    ax.plot([start_pos[0], first_order.start_lon], [start_pos[1], first_order.start_lat], 
                            color=car_color, alpha=0.7, linewidth=2, linestyle='-')
                    # 拼车接驾路径
                    prev_lon, prev_lat = first_order.start_lon, first_order.start_lat
                    for order in orders_list[1:]:
                        ax.plot([prev_lon, order.start_lon], [prev_lat, order.start_lat], 
                                color=car_color, alpha=0.7, linewidth=2, linestyle='-')
                        prev_lon, prev_lat = order.start_lon, order.start_lat
                    # 最后一个订单到终点
                    last_order = orders_list[-1]
                    ax.plot([last_order.start_lon, last_order.end_lon], [last_order.start_lat, last_order.end_lat], 
                            color=car_color, alpha=0.7, linewidth=2, linestyle='-')
                    # 标记轮次终点
                    ax.scatter(end_pos[0], end_pos[1], color=car_color, s=80, alpha=0.8, 
                               marker='s', edgecolors='white', linewidth=1)
    
    ax.set_xlabel('经度', fontsize=14)
    ax.set_ylabel('纬度', fontsize=14)
    ax.set_title('拼车订单-车辆聚类分布与接单路径', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    # 保存图片
    save_path = Path(save_dir) / "carpool_cluster_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"拼车聚类分布图已保存至：{save_path}")

def plot_carpool_statistics(subgroups: Dict[int, Tuple[List[TaxiOrder], List[NetCarLocation]]],
                            carpool_results: Dict[int, Dict[int, List[Tuple[List[TaxiOrder], Tuple[float, float], Tuple[float, float]]]]],
                            save_dir: str):
    """绘制拼车统计图表：每车接单数、总距离、拼车率"""
    # 统计数据
    car_ids = []
    total_passengers = []
    total_distances = []
    carpool_rates = []  # 拼车率=接单数/使用车辆数
    
    for subgroup_id, (_, cars) in subgroups.items():
        subgroup_matches = carpool_results[subgroup_id]
        for car_idx in range(len(cars)):
            round_matches = subgroup_matches.get(car_idx, [])
            # 总乘客数
            total_p = sum(len(orders) for orders, _, _ in round_matches)
            # 总距离
            total_d = 0.0
            for orders, start_pos, end_pos in round_matches:
                current_lon, current_lat = start_pos
                for order in orders:
                    total_d += calculate_haversine_distance(current_lon, current_lat, order.start_lon, order.start_lat)
                    total_d += calculate_haversine_distance(order.start_lon, order.start_lat, order.end_lon, order.end_lat)
                    current_lon, current_lat = order.end_lon, order.end_lat
            # 拼车率
            used_rounds = len([r for r in round_matches if len(r[0]) > 0])
            rate = total_p / used_rounds if used_rounds > 0 else 0.0
            
            car_ids.append(f'子群{subgroup_id}-车{car_idx}')
            total_passengers.append(total_p)
            total_distances.append(round(total_d, 2))
            carpool_rates.append(round(rate, 2))
    
    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. 每车接单数
    bars1 = ax1.bar(car_ids, total_passengers, color=CAR_COLORS[:len(car_ids)], alpha=0.8)
    ax1.set_xlabel('车辆ID', fontsize=12)
    ax1.set_ylabel('总接单数（人）', fontsize=12)
    ax1.set_title('每车总接单数统计', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # 2. 每车总行驶距离
    bars2 = ax2.bar(car_ids, total_distances, color=CAR_COLORS[:len(car_ids)], alpha=0.8)
    ax2.set_xlabel('车辆ID', fontsize=12)
    ax2.set_ylabel('总行驶距离（km）', fontsize=12)
    ax2.set_title('每车总行驶距离统计', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    # 添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 3. 每车拼车率
    bars3 = ax3.bar(car_ids, carpool_rates, color=CAR_COLORS[:len(car_ids)], alpha=0.8)
    ax3.set_xlabel('车辆ID', fontsize=12)
    ax3.set_ylabel('拼车率（人/轮次）', fontsize=12)
    ax3.set_title('每车拼车率统计（≤4）', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, MAX_PASSENGERS + 0.5)
    ax3.tick_params(axis='x', rotation=45)
    # 添加数值标签
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('拼车匹配统计总览', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存图片
    save_path = Path(save_dir) / "carpool_statistics.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"拼车统计图表已保存至：{save_path}")

def plot_carpool_round_details(subgroup_id: int,
                               carpool_results: Dict[int, Dict[int, List[Tuple[List[TaxiOrder], Tuple[float, float], Tuple[float, float]]]]],
                               cars: List[NetCarLocation],
                               save_dir: str):
    """绘制指定子群的多轮接单详情"""
    subgroup_matches = carpool_results[subgroup_id]
    n_cars = len([c for c in subgroup_matches.values() if c])
    n_cols = min(3, n_cars)
    n_rows = (n_cars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_cars == 1:
        axes = [axes]
    elif n_rows > 1:
        axes = axes.flatten()
    
    # 遍历车辆绘制
    car_idx_list = [cid for cid, matches in subgroup_matches.items() if matches]
    for idx, car_idx in enumerate(car_idx_list):
        ax = axes[idx] if n_cars > 1 else axes
        car_color = CAR_COLORS[car_idx % len(CAR_COLORS)]
        round_matches = subgroup_matches[car_idx]
        
        # 绘制车辆初始位置
        car = cars[car_idx]
        ax.scatter(car.lon, car.lat, color=car_color, s=120, alpha=0.9, 
                   label=f'车辆{car_idx}初始位置', marker='^', edgecolors='black', linewidth=1)
        
        # 绘制各轮次路径
        for round_idx, (orders_list, start_pos, end_pos) in enumerate(round_matches):
            # 标记轮次起点
            ax.scatter(start_pos[0], start_pos[1], color=car_color, s=80, alpha=0.8, 
                       label=f'第{round_idx+1}轮起点' if idx == 0 and round_idx == 0 else "", 
                       marker='o', edgecolors='white', linewidth=1)
            
            # 绘制接驾路径
            current_lon, current_lat = start_pos
            for order_idx, order in enumerate(orders_list):
                # 路径线
                ax.plot([current_lon, order.start_lon], [current_lat, order.start_lat], 
                        color=car_color, alpha=0.7, linewidth=2, linestyle='-')
                # 订单起点
                ax.scatter(order.start_lon, order.start_lat, color='red', s=60, alpha=0.8, 
                           marker='o', edgecolors='black', linewidth=0.5)
                # 订单终点
                ax.scatter(order.end_lon, order.end_lat, color='blue', s=60, alpha=0.8, 
                           marker='x', edgecolors='black', linewidth=0.5)
                # 送驾路径
                ax.plot([order.start_lon, order.end_lon], [order.start_lat, order.end_lat], 
                        color=car_color, alpha=0.7, linewidth=2, linestyle='--')
                
                current_lon, current_lat = order.start_lon, order.start_lat
            
            # 标记轮次终点
            ax.scatter(end_pos[0], end_pos[1], color=car_color, s=80, alpha=0.8, 
                       marker='s', edgecolors='white', linewidth=1)
        
        # 设置子图标题和标签
        total_passengers = sum(len(orders) for orders, _, _ in round_matches)
        ax.set_title(f'车辆{car_idx}接单详情\n总乘客数：{total_passengers}（{len(round_matches)}轮）', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('经度', fontsize=10)
        ax.set_ylabel('纬度', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    if n_cars < len(axes):
        for idx in range(n_cars, len(axes)):
            axes[idx].set_visible(False)
    
    plt.suptitle(f'子群{subgroup_id}多轮拼车接单详情', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图片
    save_path = Path(save_dir) / f"subgroup{subgroup_id}_carpool_round_details.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"子群{subgroup_id}拼车轮次详情图已保存至：{save_path}")

# -------------------------- 主执行逻辑 --------------------------
def main():
    """主函数：执行拼车+多轮接单PSO匹配 + 结果保存 + 可视化"""
    # 1. 生成数据并获取聚类子群
    print("===== 1. 生成数据并获取子群体 =====")
    taxi_orders = generate_taxi_orders(ORDER_NUM)
    car_locations = generate_netcar_locations(CAR_NUM)
    clustered_orders, order_kmeans, _ = cluster_taxi_orders(taxi_orders, N_CLUSTERS)
    clustered_cars, car_kmeans, _ = cluster_netcar_locations(car_locations, N_CLUSTERS)
    center_matches, _ = match_cluster_centers(order_kmeans.cluster_centers_, car_kmeans.cluster_centers_)
    subgroups = split_into_subgroups(clustered_orders, clustered_cars, center_matches)

    # 2. 子群内拼车+多轮接单匹配
    print("\n===== 2. 执行拼车+多轮接单PSO匹配 =====")
    carpool_results = {}
    for subgroup_id, (orders, cars) in subgroups.items():
        if len(orders) == 0 or len(cars) == 0:
            carpool_results[subgroup_id] = {}
            print(f"子群{subgroup_id}无订单/车辆，跳过")
            continue
        
        # 初始化拼车匹配器
        matcher = CarpoolPSOMatcher(subgroup_id, orders, cars)
        # 执行多轮匹配
        match_result = matcher.run_multi_round_matching()
        carpool_results[subgroup_id] = match_result

    # 3. 保存结果
    save_carpool_results(subgroups, carpool_results)

    # 4. 生成可视化结果
    print("\n===== 3. 生成可视化结果 =====")
    # 全局聚类+路径图
    plot_carpool_cluster_distribution(subgroups, carpool_results, STORAGE_DIR)
    # 全局统计图表
    plot_carpool_statistics(subgroups, carpool_results, STORAGE_DIR)
    # 各子群轮次详情图
    for subgroup_id in subgroups.keys():
        if len(subgroups[subgroup_id][0]) > 0:
            plot_carpool_round_details(subgroup_id, carpool_results, subgroups[subgroup_id][1], STORAGE_DIR)

    # 5. 汇总结果
    print("\n===== 4. 拼车匹配结果汇总 =====")
    total_orders = sum(len(orders) for orders, _ in subgroups.values())
    total_matched = 0
    total_distance = 0.0
    total_cars_used = 0
    total_passengers = 0

    for subgroup_id, (orders, cars) in subgroups.items():
        subgroup_matches = carpool_results[subgroup_id]
        for car_idx, round_matches in subgroup_matches.items():
            if round_matches:
                total_cars_used += 1
                for orders_list, start_pos, end_pos in round_matches:
                    total_passengers += len(orders_list)
                    # 计算总距离
                    current_lon, current_lat = start_pos
                    for order in orders_list:
                        total_distance += calculate_haversine_distance(current_lon, current_lat, order.start_lon, order.start_lat)
                        total_distance += calculate_haversine_distance(order.start_lon, order.start_lat, order.end_lon, order.end_lat)
                        current_lon, current_lat = order.end_lon, order.end_lat
        total_matched += len(orders) - len([o for o in orders if not any(o in r[0] for _, round_matches in subgroup_matches.items() for r in round_matches)])

    print(f"全局总订单数：{total_orders}")
    print(f"全局已匹配订单数：{total_matched}（完成率：{total_matched/total_orders*100:.2f}%）")
    print(f"全局总乘客数：{total_passengers}（拼车率：{total_passengers/total_cars_used:.2f}人/车）")
    print(f"全局总行驶距离：{total_distance:.2f}km")
    print(f"平均每单行驶距离：{total_distance/total_matched:.2f}km")

if __name__ == "__main__":
    main()

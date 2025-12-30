import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, DefaultDict
from collections import defaultdict
import matplotlib.cm as cm
from matplotlib.path import Path
import matplotlib.patches as patches
from Order_generate import TaxiOrder, generate_taxi_orders, ORDER_NUM
from Car_generate import NetCarLocation, generate_netcar_locations, CAR_NUM
from Distance_transfer import cal_km_by_lon_lat
from K_means import TaxiCarClusterMatcher

class PSOOrderMatcher:
    """粒子群优化订单匹配器（含全部可视化+完整性能指标计算功能）"""
    
    def __init__(
        self,
        subgroup_orders: List[TaxiOrder],
        subgroup_cars: List[NetCarLocation],
        w: float = 0.5,  # PSO惯性权重
        c1: float = 1.0, # 认知系数
        c2: float = 1.0, # 社会系数
        max_iter: int = 50,
        pop_size: int = 30,
        # 多目标归一化参数
        k_distance: float = 0.001,
        k_imbalance: float = 0.5,
        k_carpool: float = 0.8,
        k_unassigned: float = 1.0,
        # 多目标权重
        weight_distance: float = 0.5,
        weight_imbalance: float = 0.2,
        weight_carpool: float = 0.2,
        weight_unassigned: float = 0.1,
        empty_weight: float = 1.5
    ):
        self.orders = subgroup_orders
        self.cars = subgroup_cars
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.pop_size = pop_size
        
        # 多目标参数
        self.k_distance = k_distance
        self.k_imbalance = k_imbalance
        self.k_carpool = k_carpool
        self.k_unassigned = k_unassigned
        
        self.weight_distance = weight_distance
        self.weight_imbalance = weight_imbalance
        self.weight_carpool = weight_carpool
        self.weight_unassigned = weight_unassigned
        self.empty_weight = empty_weight
        
        # 车辆状态
        self.car_states = {
            car.car_id: {
                "current_lon": car.lon,
                "current_lat": car.lat,
                "assigned_orders": [],
                "total_passengers": 0
            } for car in self.cars
        }
        
        # PSO核心参数
        self.particles = []
        self.pbest = []
        self.gbest = None
        self.pbest_fitness = []
        self.gbest_fitness = float('inf')
        
        # 优化过程记录（用于可视化）
        self.gbest_fitness_history = []  # 记录每代的全局最优适应度
        
        # 基础统计值
        self.order_ids = [o.order_id for o in self.orders]
        self.car_ids = [c.car_id for c in self.cars]
        self.n_orders = len(self.orders)
        self.n_cars = len(self.cars)
        self.order_id_map = {o.order_id: o for o in self.orders}
        self.avg_orders_per_car = self.n_orders / self.n_cars if self.n_cars > 0 else 0
        
        # 预计算参考值
        self.max_distance_ref = self._calc_max_distance_ref()
        self.max_imbalance_ref = self.n_orders
        self.max_carpool_ref = 4
        self.max_unassigned_ref = self.n_orders
        
        # ========== 新增：性能指标存储 ==========
        self.performance_metrics = {}

    def _calc_max_distance_ref(self) -> float:
        if self.n_orders == 0:
            return 1.0
        total = 0.0
        for order in self.orders:
            if not self.car_ids:
                continue
            empty_max = cal_km_by_lon_lat(
                self.car_states[self.car_ids[0]]["current_lon"],
                self.car_states[self.car_ids[0]]["current_lat"],
                order.start_lon, order.start_lat
            ) * 2
            ride = cal_km_by_lon_lat(
                order.start_lon, order.start_lat,
                order.end_lon, order.end_lat
            )
            total += empty_max * self.empty_weight + ride
        return total

    def _normalize(self, x: float, k: float, max_ref: float) -> float:
        if max_ref == 0 or x <= 0:
            return 0.0
        normalized_x = x / max_ref
        return 1 - np.exp(-k * normalized_x)

    def _initialize_particles(self):
        for _ in range(self.pop_size):
            particle = {o.order_id: np.random.choice(self.car_ids) for o in self.orders}
            self.particles.append(particle)
            self.pbest.append(particle.copy())
            self.pbest_fitness.append(float('inf'))

    def _calculate_fitness(self, particle: Dict[str, str]) -> float:
        total_weighted_distance = 0.0
        total_imbalance = 0.0
        total_carpool_exceed = 0.0
        total_unassigned = 0.0

        car_order_map = defaultdict(list)
        for order_id, car_id in particle.items():
            car_order_map[car_id].append(order_id)
        
        temp_car_states = {k: {**v} for k, v in self.car_states.items()}
        assigned_orders = set()

        for car_id, order_ids in car_order_map.items():
            car_state = temp_car_states[car_id]
            current_lon, current_lat = car_state["current_lon"], car_state["current_lat"]
            current_passengers = 0
            car_assigned = []

            for order_id in order_ids:
                order = self.order_id_map[order_id]
                empty_dist = cal_km_by_lon_lat(
                    current_lon, current_lat,
                    order.start_lon, order.start_lat
                )
                total_weighted_distance += empty_dist * self.empty_weight
                ride_dist = cal_km_by_lon_lat(
                    order.start_lon, order.start_lat,
                    order.end_lon, order.end_lat
                )
                total_weighted_distance += ride_dist

                if order.is_carpool:
                    exceed = max(0, current_passengers + order.passenger_num - 4)
                    total_carpool_exceed += exceed
                    if exceed == 0:
                        current_passengers += order.passenger_num
                        car_assigned.append(order_id)
                        assigned_orders.add(order_id)
                else:
                    current_passengers = order.passenger_num
                    car_assigned.append(order_id)
                    assigned_orders.add(order_id)

                current_lon, current_lat = order.end_lon, order.end_lat

            imbalance = max(0, len(car_assigned) - self.avg_orders_per_car)
            total_imbalance += imbalance

        total_unassigned = self.n_orders - len(assigned_orders)

        norm_distance = self._normalize(
            total_weighted_distance, self.k_distance, self.max_distance_ref
        )
        norm_imbalance = self._normalize(
            total_imbalance, self.k_imbalance, self.max_imbalance_ref
        )
        norm_carpool = self._normalize(
            total_carpool_exceed, self.k_carpool, self.max_carpool_ref
        )
        norm_unassigned = self._normalize(
            total_unassigned, self.k_unassigned, self.max_unassigned_ref
        )

        total_fitness = (
            norm_distance * self.weight_distance +
            norm_imbalance * self.weight_imbalance +
            norm_carpool * self.weight_carpool +
            norm_unassigned * self.weight_unassigned
        )

        return total_fitness

    def _update_velocity_position(self, particle_idx: int):
        current_p = self.particles[particle_idx]
        pbest_p = self.pbest[particle_idx]

        for order_id in self.order_ids:
            r1, r2 = np.random.random(), np.random.random()
            cog_prob = self.w * self.c1 * r1
            soc_prob = self.w * self.c2 * r2

            if cog_prob > 0.5:
                current_p[order_id] = pbest_p[order_id]
            if soc_prob > 0.5 and self.gbest is not None:
                current_p[order_id] = self.gbest[order_id]

    def optimize(self) -> Tuple[Dict[str, str], float]:
        """执行PSO优化，返回最优匹配和适应度，同时记录优化过程"""
        self._initialize_particles()
        self.gbest_fitness_history = []  # 重置历史记录

        for _ in range(self.max_iter):
            current_gbest = float('inf')
            current_gbest_particle = None
            
            # 计算所有粒子的适应度
            for i in range(self.pop_size):
                fitness = self._calculate_fitness(self.particles[i])
                # 更新个体最优
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest[i] = self.particles[i].copy()
                # 跟踪当前代的最优
                if fitness < current_gbest:
                    current_gbest = fitness
                    current_gbest_particle = self.particles[i].copy()
            
            # 更新全局最优
            if current_gbest < self.gbest_fitness:
                self.gbest_fitness = current_gbest
                self.gbest = current_gbest_particle
            
            # 记录当前代的全局最优适应度
            self.gbest_fitness_history.append(self.gbest_fitness)
            
            # 更新粒子位置
            for i in range(self.pop_size):
                self._update_velocity_position(i)
        
        # ========== 优化完成后，立即计算所有性能指标 ==========
        self.calculate_performance_metrics(self.gbest)
        return self.gbest, self.gbest_fitness

    # ======================== 核心新增：6类性能指标完整计算方法 ========================
    def calculate_performance_metrics(self, best_matching: Dict[str, str]) -> Dict[str, float]:
        """计算所有核心性能指标，返回指标字典，指标全部量化，数值越小/越高越好（标注）"""
        car_order_map = defaultdict(list)
        for o_id, c_id in best_matching.items():
            car_order_map[c_id].append(o_id)
        
        # 初始化统计变量
        total_empty_distance = 0.0    # 总空驶距离 (车辆→订单起点)
        total_ride_distance = 0.0     # 总载客距离 (订单起点→终点)
        total_all_distance = 0.0      # 总行驶距离 = 空驶+载客
        car_order_counts = defaultdict(int) # 每辆车分配的订单数
        car_distances = defaultdict(float)  # 每辆车的总行驶距离
        unassigned_order_num = self.n_orders - len(best_matching) # 未分配订单数
        carpool_overload = 0          # 拼车超载次数
        assigned_orders = set(best_matching.keys())

        # 遍历每辆车，计算路径距离+约束指标
        for car_id, order_ids in car_order_map.items():
            car = next(c for c in self.cars if c.car_id == car_id)
            curr_lon, curr_lat = car.lon, car.lat
            curr_passengers = 0
            car_total_dist = 0.0
            car_order_counts[car_id] = len(order_ids)

            for o_id in order_ids:
                order = self.order_id_map[o_id]
                # 计算空驶距离：当前位置 → 订单起点
                empty_dist = cal_km_by_lon_lat(curr_lon, curr_lat, order.start_lon, order.start_lat)
                # 计算载客距离：订单起点 → 订单终点
                ride_dist = cal_km_by_lon_lat(order.start_lon, order.start_lat, order.end_lon, order.end_lat)
                
                # 累加距离
                total_empty_distance += empty_dist
                total_ride_distance += ride_dist
                car_total_dist += empty_dist + ride_dist
                
                # 拼车超载检查
                if order.is_carpool:
                    if curr_passengers + order.passenger_num > 4:
                        carpool_overload +=1
                    else:
                        curr_passengers += order.passenger_num
                else:
                    curr_passengers = order.passenger_num
                
                # 更新车辆当前位置（核心逻辑：上一单终点作为下一单起点）
                curr_lon, curr_lat = order.end_lon, order.end_lat
            
            car_distances[car_id] = car_total_dist
            total_all_distance += car_total_dist
        
        # ========== 指标1：成本类（核心）- 越小越好 ==========
        avg_empty_distance = total_empty_distance / self.n_orders if self.n_orders>0 else 0  # 订单平均空驶距离
        avg_all_distance = total_all_distance / self.n_orders if self.n_orders>0 else 0        # 订单平均总行驶距离
        empty_ratio = total_empty_distance / total_all_distance if total_all_distance>0 else 0 # 空驶率(空驶/总距离)
        
        # ========== 指标2：负载均衡类（核心）- 越小越好 ==========
        order_counts = list(car_order_counts.values())
        std_order = np.std(order_counts) if len(order_counts)>0 else 0 # 订单数标准差（越小越均衡）
        cv_order = std_order / self.avg_orders_per_car if self.avg_orders_per_car>0 else 0 # 变异系数（标准化的均衡指标）
        
        # ========== 指标3：服务质量类（硬性约束）- 越小越好 ==========
        unassigned_rate = unassigned_order_num / self.n_orders if self.n_orders>0 else 0 # 订单未分配率
        overload_rate = carpool_overload / self.n_orders if self.n_orders>0 else 0       # 拼车超载率
        
        # ========== 指标4：资源利用率类（核心）- 越高越好 ==========
        car_utilization = np.mean([1 if cnt>0 else 0 for cnt in order_counts]) if len(order_counts)>0 else 0 # 车辆利用率
        avg_car_distance = total_all_distance / self.n_cars if self.n_cars>0 else 0                        # 车辆平均行驶距离
        
        # ========== 指标5：收敛性指标（算法效率）- 越小越好 ==========
        final_fitness = self.gbest_fitness
        converge_iter = len(self.gbest_fitness_history) # 收敛迭代次数
        
        # ========== 指标6：综合调度效率（核心）- 越小越好 ==========
        avg_order_service_dist = avg_all_distance # 订单平均服务距离（等价于平均总行驶距离）

        # 存储所有指标
        self.performance_metrics = {
            # 成本类 (越小越好)
            '总行驶距离(km)': round(total_all_distance, 2),
            '空驶率(%)': round(empty_ratio * 100, 2),
            '订单平均空驶距离(km)': round(avg_empty_distance, 2),
            # 均衡类 (越小越好)
            '订单数标准差': round(std_order, 2),
            '订单数变异系数': round(cv_order, 2),
            # 服务质量 (越小越好)
            '订单未分配率(%)': round(unassigned_rate * 100, 2),
            '拼车超载率(%)': round(overload_rate * 100, 2),
            # 利用率 (越高越好)
            '车辆利用率(%)': round(car_utilization * 100, 2),
            # 算法收敛 (越小越好)
            '最优适应度值': round(final_fitness, 4),
            '收敛迭代次数': converge_iter
        }
        return self.performance_metrics

    # ======================== 原有可视化方法 不变 ========================
    def plot_pso_optimization_process(self, figsize: Tuple[int, int] = (10, 6)):
        if not self.gbest_fitness_history:
            raise RuntimeError("请先执行优化操作（optimize）")
        plt.figure(figsize=figsize)
        plt.plot(range(1, self.max_iter + 1), self.gbest_fitness_history, 'b-', linewidth=2)
        plt.title('PSO Optimization Process - Best Fitness Over Iterations', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Fitness Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(range(0, self.max_iter + 1, max(1, self.max_iter // 10)))
        plt.tight_layout()
        plt.show()

    def plot_vehicle_load_balance(self, best_matching: Dict[str, str], figsize: Tuple[int, int] = (12, 6)):
        if not best_matching:
            raise RuntimeError("请先获取最优匹配结果")
        car_order_count = defaultdict(int)
        for _, cid in best_matching.items():
            car_order_count[cid] += 1
        cars = list(car_order_count.keys())
        counts = list(car_order_count.values())
        avg_line = [self.avg_orders_per_car] * len(cars)
        plt.figure(figsize=figsize)
        x = np.arange(len(cars))
        bars = plt.bar(x, counts, width=0.6, label='Actual Orders')
        plt.plot(x, avg_line, 'r--', linewidth=2, label=f'Theoretical Average ({self.avg_orders_per_car:.2f})')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height}', ha='center', va='bottom')
        plt.title('Vehicle Load Balance', fontsize=14)
        plt.xlabel('Vehicle ID', fontsize=12)
        plt.ylabel('Number of Assigned Orders', fontsize=12)
        plt.xticks(x, cars, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

# ======================== 原有绘图函数 不变 ========================
def plot_subgroup_matching(subgroup_id: int, orders: List[TaxiOrder], cars: List[NetCarLocation], matching: Dict[str, str], figsize: Tuple[int, int] = (12, 10)):
    if not orders or not cars or not matching:
        raise ValueError("订单、车辆或匹配结果不能为空")
    car_ids = [car.car_id for car in cars]
    colors = cm.rainbow(np.linspace(0, 1, len(car_ids)))
    car_color_map = {car_id: color for car_id, color in zip(car_ids, colors)}
    plt.figure(figsize=figsize)
    order_starts_x = [order.start_lon for order in orders]
    order_starts_y = [order.start_lat for order in orders]
    plt.scatter(order_starts_x, order_starts_y, c='blue', s=80, marker='o', label='Order Start')
    order_ends_x = [order.end_lon for order in orders]
    order_ends_y = [order.end_lat for order in orders]
    plt.scatter(order_ends_x, order_ends_y, c='green', s=80, marker='s', label='Order End')
    car_positions = {}
    for car in cars:
        x, y = car.lon, car.lat
        car_positions[car.car_id] = (x, y)
        plt.scatter(x, y, c=[car_color_map[car.car_id]], s=150, marker='^', edgecolors='black', linewidth=1.5)
        plt.text(x, y, f" {car.car_id.split()[1]}", fontsize=9)
    for order_id, car_id in matching.items():
        order = next(o for o in orders if o.order_id == order_id)
        order_x, order_y = order.start_lon, order.start_lat
        car_x, car_y = car_positions[car_id]
        plt.plot([order_x, car_x], [order_y, car_y], color=car_color_map[car_id], linestyle='-', alpha=0.6, linewidth=1.5)
    plt.title(f'Subgroup {subgroup_id} - Order-Vehicle Matching', fontsize=16)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_directed_routes(matcher: PSOOrderMatcher, best_matching: Dict[str, str], subgroup_id: int, figsize: Tuple[int, float] = (12, 8)):
    if not matcher.orders or not matcher.cars or not best_matching:
        raise ValueError("数据不能为空")
    car_routes = {}
    car_order_map = defaultdict(list)
    for order_id, car_id in best_matching.items():
        car_order_map[car_id].append(order_id)
    for car_id, order_ids in car_order_map.items():
        car = next(c for c in matcher.cars if c.car_id == car_id)
        route = [(car.lon, car.lat)]
        current_lon, current_lat = car.lon, car.lat
        for order_id in order_ids:
            order = matcher.order_id_map[order_id]
            route.append((order.start_lon, order.start_lat))
            route.append((order.end_lon, order.end_lat))
            current_lon, current_lat = order.end_lon, order.end_lat
        car_routes[car_id] = route
    plt.figure(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(car_routes)))
    for idx, (car_id, route) in enumerate(car_routes.items()):
        color = colors[idx]
        coords = np.array(route)
        plt.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2, alpha=0.8, label=f'Car {car_id.split()[1]}')
        for i in range(len(coords) - 1):
            start_x, start_y = coords[i]
            end_x, end_y = coords[i+1]
            dx = end_x - start_x
            dy = end_y - start_y
            plt.arrow(start_x, start_y, dx, dy, head_width=0.003, head_length=0.006, fc=color, ec=color, linewidth=1, length_includes_head=True)
        plt.scatter(coords[0,0], coords[0,1], c=color, s=200, marker='^', edgecolors='black', zorder=5)
        plt.scatter(coords[-1,0], coords[-1,1], c=color, s=150, marker='o', facecolors='none', edgecolors='black', zorder=5)
    for order in matcher.orders:
        plt.scatter(order.start_lon, order.start_lat, c='darkblue', s=60, marker='s', label='Order Start' if order.order_id == matcher.orders[0].order_id else "")
        plt.scatter(order.end_lon, order.end_lat, c='darkgreen', s=60, marker='D', label='Order End' if order.order_id == matcher.orders[0].order_id else "")
    plt.title(f'Subgroup {subgroup_id} - Vehicle Service Route (Directed Path)', fontsize=15)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    plt.show()

# ======================== 新增：性能指标可视化函数 ========================
def plot_performance_metrics(metrics: Dict[str, float], subgroup_id: int, figsize: Tuple[int, int] = (14, 8)):
    """可视化核心性能指标，柱状图+数值标注，直观展示优劣"""
    plt.figure(figsize=figsize)
    # 筛选核心可视化指标
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    # 配色：越小越好的指标标蓝色，越高越好的标绿色
    colors = ['#1f77b4' if name not in ['车辆利用率(%)'] else '#2ca02c' for name in metric_names]
    x = np.arange(len(metric_names))
    bars = plt.bar(x, metric_values, color=colors, alpha=0.8, width=0.6)
    # 添加数值标注
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{val}', ha='center', va='bottom', fontsize=10)
    # 图表配置
    plt.title(f'Subgroup {subgroup_id} - Algorithm Performance Metrics', fontsize=15)
    plt.xlabel('Performance Metrics', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.xticks(x, metric_names, rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ======================== 主函数 ========================
def main():
    # 生成数据
    orders = generate_taxi_orders(ORDER_NUM)
    cars = generate_netcar_locations(CAR_NUM)
    print(f"✅ 生成数据完成：{len(orders)}个订单，{len(cars)}辆网约车")

    # K-means聚类分群
    cluster_matcher = TaxiCarClusterMatcher(n_clusters=4)
    cluster_matcher.cluster_taxi_orders(orders)
    cluster_matcher.cluster_netcar_locations(cars)
    cluster_matcher.match_cluster_centers()
    subgroups = cluster_matcher.split_into_subgroups()

    # 子群PSO匹配及全流程可视化+指标输出
    all_results = {}
    for subgroup_id, (sub_orders, sub_cars) in subgroups.items():
        print(f"\n{'='*60}")
        print(f"✅ 处理子群体 {subgroup_id}：{len(sub_orders)}订单 | {len(sub_cars)}车辆")
        if not sub_orders or not sub_cars:
            print("❌ 子群无数据，跳过")
            continue

        # 初始化PSO匹配器
        pso_matcher = PSOOrderMatcher(
            sub_orders, sub_cars,
            w=0.7, c1=1.2, c2=1.2,
            max_iter=100, pop_size=50,
            k_distance=0.001, k_imbalance=0.5, k_carpool=0.8, k_unassigned=1.0,
            weight_distance=0.5, weight_imbalance=0.2, weight_carpool=0.2, weight_unassigned=0.1,
            empty_weight=1.5
        )
        best_matching, best_fitness = pso_matcher.optimize()

        # ========== 1. 打印量化性能指标（核心，优先看这个） ==========
        print(f"\n📊 子群体 {subgroup_id} 性能指标量化结果（核心）：")
        for metric_name, metric_val in pso_matcher.performance_metrics.items():
            print(f"   {metric_name}: {metric_val}")

        # ========== 2. 依次执行所有可视化 ==========
        plot_subgroup_matching(subgroup_id, sub_orders, sub_cars, best_matching)    # 匹配连线图
        pso_matcher.plot_pso_optimization_process()                                # PSO收敛曲线
        pso_matcher.plot_vehicle_load_balance(best_matching)                       # 负载均衡柱状图
        plot_directed_routes(pso_matcher, best_matching, subgroup_id)              # 有向路径图
        plot_performance_metrics(pso_matcher.performance_metrics, subgroup_id)     # 性能指标可视化图

        # 统计结果
        car_order_count = defaultdict(int)
        for oid, cid in best_matching.items():
            car_order_count[cid] += 1
        avg_actual = sum(car_order_count.values()) / len(car_order_count) if car_order_count else 0

        all_results[subgroup_id] = {
            "best_matching": best_matching,
            "best_fitness": best_fitness,
            "metrics": pso_matcher.performance_metrics,
            "avg_orders_theory": pso_matcher.avg_orders_per_car,
            "avg_orders_actual": avg_actual
        }

        print(f"\n✅ 子群体 {subgroup_id} 调度完成 | 最优适应度：{best_fitness:.4f}（越小越好）")

if __name__ == "__main__":
    main()

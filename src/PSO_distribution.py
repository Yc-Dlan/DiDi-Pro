import numpy as np
from typing import List, Dict, Tuple, Optional, DefaultDict
from collections import defaultdict
from Order_generate import TaxiOrder, generate_taxi_orders, ORDER_NUM
from Car_generate import NetCarLocation, generate_netcar_locations, CAR_NUM
from Distance_transfer import cal_km_by_lon_lat
from K_means import TaxiCarClusterMatcher

class PSOOrderMatcher:
    """粒子群优化订单匹配器"""
    def __init__(
        self,
        subgroup_orders,
        subgroup_cars,
        w = 0.5,  # PSO惯性权重
        c1 = 1.0, # 认知系数
        c2 = 1.0, # 社会系数
        max_iter = 50,
        pop_size = 30,
        # 多目标归一化参数
        k_distance = 0.001,  # 路程目标的k
        k_imbalance = 0.5,    # 分配不均的k
        k_carpool = 0.8,      # 拼车超限的k
        k_unassigned = 1.0,   # 未分配订单的k
        # 多目标权重
        weight_distance= 0.5,  # 路程权重
        weight_imbalance = 0.2, # 分配不均权重
        weight_carpool = 0.2,   # 拼车超限权重
        weight_unassigned = 0.1,# 未分配订单权重
        empty_weight = 1.5      # 空驶路程的额外权重
    ):
        self.orders = subgroup_orders
        self.cars = subgroup_cars
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.pop_size = pop_size
        
        # 多目标归一化参数
        self.k_distance = k_distance
        self.k_imbalance = k_imbalance
        self.k_carpool = k_carpool
        self.k_unassigned = k_unassigned
        
        # 多目标权重
        self.weight_distance = weight_distance
        self.weight_imbalance = weight_imbalance
        self.weight_carpool = weight_carpool
        self.weight_unassigned = weight_unassigned
        self.empty_weight = empty_weight
        
        # 车辆状态初始化
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
        
        # 基础统计值
        self.order_ids = [o.order_id for o in self.orders]
        self.car_ids = [c.car_id for c in self.cars]
        self.n_orders = len(self.orders)
        self.n_cars = len(self.cars)
        self.order_id_map = {o.order_id: o for o in self.orders}
        self.avg_orders_per_car = self.n_orders / self.n_cars if self.n_cars > 0 else 0
        
        # 预计算各目标的最大参考值
        self.max_distance_ref = self._calc_max_distance_ref()  # 路程最大参考值
        self.max_imbalance_ref = self.n_orders  # 分配不均最大参考值
        self.max_carpool_ref = 4  # 拼车超限最大参考值
        self.max_unassigned_ref = self.n_orders  # 未分配订单最大参考值

    def _calc_max_distance_ref(self):
        """计算路程最大参考值"""
        if self.n_orders == 0:
            return 1.0
        total = 0.0
        for order in self.orders:
            # 空驶:简化为两倍实际距离
            empty_max = cal_km_by_lon_lat(
                self.car_states[self.car_ids[0]]["current_lon"],
                self.car_states[self.car_ids[0]]["current_lat"],
                order.start_lon, order.start_lat
            ) * 2
            # 载客:订单本身的距离
            ride = cal_km_by_lon_lat(
                order.start_lon, order.start_lat,
                order.end_lon, order.end_lat
            )
            total += empty_max * self.empty_weight + ride
        return total

    def _normalize(self, x, k, max_ref):
        """
        归一化函数 fx = 1 - e^(-k * (x / max_ref))
        - x: 原始变量值（路程/分配不均数/超限数/未分配数）
        - k: 该目标的惩罚系数
        - max_ref: 该目标的最大参考值(校准x的量级)
        - return: 归一化后的值
        """
        if max_ref == 0 or x <= 0:
            return 0.0
        normalized_x = x / max_ref  
        return 1 - np.exp(-k * normalized_x)

    def _initialize_particles(self):
        """初始化粒子群：订单→车辆的分配方案"""
        for _ in range(self.pop_size):
            particle = {o.order_id: np.random.choice(self.car_ids) for o in self.orders}
            self.particles.append(particle)
            self.pbest.append(particle.copy())
            self.pbest_fitness.append(float('inf'))

    def _calculate_fitness(self, particle):
        """
        多目标加权适应度计算：
        适应度 = 归一化路程×路程权重 + 归一化分配不均×不均权重 + 归一化拼车超限×拼车权重 + 归一化未分配×未分配权重
        """
        # 路程相关：空驶（加权）+ 载客
        total_weighted_distance = 0.0
        # 分配不均：所有车辆超出平均订单数的总和
        total_imbalance = 0.0
        # 拼车超限：所有车辆拼车乘客超限的总和
        total_carpool_exceed = 0.0
        # 未分配订单数
        total_unassigned = 0.0

        # 按车辆分组订单
        car_order_map = defaultdict(list)
        for order_id, car_id in particle.items():
            car_order_map[car_id].append(order_id)
        
        temp_car_states = {k: {**v} for k, v in self.car_states.items()}
        assigned_orders = set()

        # 遍历每辆车，计算接送顺序的路程+约束违反值
        for car_id, order_ids in car_order_map.items():
            car_state = temp_car_states[car_id]
            current_lon, current_lat = car_state["current_lon"], car_state["current_lat"]
            current_passengers = 0
            car_assigned = []

            # 按接送顺序计算路程
            for order_id in order_ids:
                order = self.order_id_map[order_id]
                # 空驶路程
                empty_dist = cal_km_by_lon_lat(
                    current_lon, current_lat,
                    order.start_lon, order.start_lat
                )
                total_weighted_distance += empty_dist * self.empty_weight
                # 载客路程
                ride_dist = cal_km_by_lon_lat(
                    order.start_lon, order.start_lat,
                    order.end_lon, order.end_lat
                )
                total_weighted_distance += ride_dist

                # 拼车超限检查
                if order.is_carpool:
                    exceed = max(0, current_passengers + order.passenger_num - 4)
                    total_carpool_exceed += exceed
                    if exceed == 0:
                        current_passengers += order.passenger_num
                        car_assigned.append(order_id)
                        assigned_orders.add(order_id)
                else:
                    # 非拼车独占车辆
                    current_passengers = order.passenger_num
                    car_assigned.append(order_id)
                    assigned_orders.add(order_id)

                # 更新车辆位置（接送顺序）
                current_lon, current_lat = order.end_lon, order.end_lat

            # 计算该车辆的分配不均值（仅惩罚超出平均的部分）
            imbalance = max(0, len(car_assigned) - self.avg_orders_per_car)
            total_imbalance += imbalance

        # 计算未分配订单数
        total_unassigned = self.n_orders - len(assigned_orders)

        # ========== 对每个目标单独归一化 ==========
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

        # ========== 多目标加权求和得到总适应度 ==========
        total_fitness = (
            norm_distance * self.weight_distance +
            norm_imbalance * self.weight_imbalance +
            norm_carpool * self.weight_carpool +
            norm_unassigned * self.weight_unassigned
        )

        return total_fitness

    def _update_velocity_position(self, particle_idx):
        """更新粒子位置"""
        current_p = self.particles[particle_idx]
        pbest_p = self.pbest[particle_idx]

        for order_id in self.order_ids:
            r1, r2 = np.random.random(), np.random.random()
            # 融入惯性权重的概率计算
            cog_prob = self.w * self.c1 * r1
            soc_prob = self.w * self.c2 * r2

            if cog_prob > 0.5:
                current_p[order_id] = pbest_p[order_id]
            if soc_prob > 0.5 and self.gbest is not None:
                current_p[order_id] = self.gbest[order_id]

    def optimize(self):
        """执行PSO优化,返回最优匹配和适应度"""
        self._initialize_particles()

        for _ in range(self.max_iter):
            # 计算所有粒子的适应度并更新最优
            for i in range(self.pop_size):
                fitness = self._calculate_fitness(self.particles[i])
                # 更新个体最优
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest[i] = self.particles[i].copy()
                # 更新全局最优
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest = self.particles[i].copy()
            # 更新粒子位置
            for i in range(self.pop_size):
                self._update_velocity_position(i)

        return self.gbest, self.gbest_fitness

def main():
    # 生成数据
    orders = generate_taxi_orders(ORDER_NUM)
    cars = generate_netcar_locations(CAR_NUM)
    print(f"生成数据：{len(orders)}订单，{len(cars)}车辆")

    # K-means聚类分群
    cluster_matcher = TaxiCarClusterMatcher(n_clusters=4)
    cluster_matcher.cluster_taxi_orders(orders)
    cluster_matcher.cluster_netcar_locations(cars)
    cluster_matcher.match_cluster_centers()
    subgroups = cluster_matcher.split_into_subgroups()

    # 子群PSO匹配
    all_results = {}
    for subgroup_id, (sub_orders, sub_cars) in subgroups.items():
        print(f"\n处理子群 {subgroup_id}:{len(sub_orders)}订单，{len(sub_cars)}车辆")
        if not sub_orders or not sub_cars:
            print("子群无订单/车辆，跳过")
            continue

        # 初始化PSO匹配器
        pso_matcher = PSOOrderMatcher(
            sub_orders, sub_cars,
            w=0.7, c1=1.2, c2=1.2,
            max_iter=100, pop_size=50,
            # 归一化k值
            k_distance=0.001, k_imbalance=0.5, k_carpool=0.8, k_unassigned=1.0,
            # 多目标权重
            weight_distance=0.5, weight_imbalance=0.2, weight_carpool=0.2, weight_unassigned=0.1,
            empty_weight=1.5  # 空驶路程权重
        )
        best_matching, best_fitness = pso_matcher.optimize()

        # 统计结果
        car_order_count = defaultdict(int)
        for oid, cid in best_matching.items():
            car_order_count[cid] += 1
        avg_actual = sum(car_order_count.values()) / len(car_order_count) if car_order_count else 0

        all_results[subgroup_id] = {
            "best_matching": best_matching,
            "best_fitness": best_fitness,
            "avg_orders_theory": pso_matcher.avg_orders_per_car,
            "avg_orders_actual": avg_actual,
            "imbalance_total": sum(max(0, cnt - pso_matcher.avg_orders_per_car) for cnt in car_order_count.values())
        }

        # 打印结果
        print(f"最优适应度：{best_fitness:.4f}（越小越好）")
        print(f"理论平均订单数：{pso_matcher.avg_orders_per_car:.2f} | 实际平均：{avg_actual:.2f}")
        print(f"总分配不均值：{all_results[subgroup_id]['imbalance_total']:.2f}")
        for cid, cnt in car_order_count.items():
            print(f"车辆 {cid} 分配订单数：{cnt}（超出平均：{cnt - pso_matcher.avg_orders_per_car:.2f}）")

if __name__ == "__main__":
    main()

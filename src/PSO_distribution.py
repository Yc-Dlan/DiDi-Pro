import numpy as np
from typing import List, Dict, Tuple, Optional
from Order_generate import TaxiOrder, generate_taxi_orders, ORDER_NUM, cal_km_by_lon_lat
from Car_generate import NetCarLocation, generate_netcar_locations, CAR_NUM
from K_means import TaxiCarClusterMatcher

class PSOOrderMatcher:
    """粒子群优化订单匹配器，处理子群体内的订单-车辆匹配"""
    
    def __init__(
        self,
        subgroup_orders: List[TaxiOrder],
        subgroup_cars: List[NetCarLocation],
        w: float = 0.5,  # 惯性权重
        c1: float = 1.0, # 认知系数
        c2: float = 1.0, # 社会系数
        max_iter: int = 50,
        pop_size: int = 30
    ):
        self.orders = subgroup_orders
        self.cars = subgroup_cars
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.pop_size = pop_size
        
        # 初始化车辆状态：当前位置（初始为车辆初始位置）和已分配订单
        self.car_states = {
            car.car_id: {
                "current_lon": car.lon,
                "current_lat": car.lat,
                "assigned_orders": [],
                "total_passengers": 0
            } for car in self.cars
        }
        
        # 粒子群相关参数
        self.particles = []  # 粒子群：每个粒子是一种分配方案
        self.pbest = []      # 个体最优
        self.gbest = None    # 全局最优
        self.pbest_fitness = []
        self.gbest_fitness = float('inf')
        
        self.order_ids = [order.order_id for order in self.orders]
        self.car_ids = [car.car_id for car in self.cars]
        self.n_orders = len(self.orders)

    def _initialize_particles(self):
        """初始化粒子群：每个粒子表示订单到车辆的分配关系"""
        for _ in range(self.pop_size):
            particle = {}
            # 随机分配订单到车辆
            for order in self.orders:
                car_id = np.random.choice(self.car_ids)
                particle[order.order_id] = car_id
            self.particles.append(particle)
            self.pbest.append(particle.copy())
            self.pbest_fitness.append(float('inf'))

    def _calculate_fitness(self, particle: Dict[str, str]) -> float:
        """计算适应度：总行驶距离（越小越好），包含约束惩罚"""
        # 深拷贝车辆状态用于模拟分配
        temp_car_states = {k: {**v} for k, v in self.car_states.items()}
        total_distance = 0.0
        
        # 按粒子分配方案处理订单
        for order_id, car_id in particle.items():
            order = next(o for o in self.orders if o.order_id == order_id)
            car_state = temp_car_states[car_id]
            
            # 计算车辆当前位置到订单起点的距离
            start_dist = cal_km_by_lon_lat(
                car_state["current_lon"], car_state["current_lat"],
                order.start_lon, order.start_lat
            )
            
            # 计算订单起点到终点的距离
            order_dist = cal_km_by_lon_lat(
                order.start_lon, order.start_lat,
                order.end_lon, order.end_lat
            )
            
            total_distance += start_dist + order_dist
            
            # 处理拼车和车辆状态更新
            if order.is_carpool:
                # 检查拼车可行性（总人数≤4）
                if car_state["total_passengers"] + order.passenger_num <= 4:
                    car_state["total_passengers"] += order.passenger_num
                    car_state["assigned_orders"].append(order_id)
                else:
                    # 拼车人数超限，添加惩罚
                    total_distance += 1000  # 惩罚值可调整
            else:
                # 非拼车订单：清空当前乘客，更新位置到终点
                car_state["total_passengers"] = order.passenger_num
                car_state["assigned_orders"] = [order_id]
            
            # 更新车辆当前位置为订单终点
            car_state["current_lon"] = order.end_lon
            car_state["current_lat"] = order.end_lat
        
        # 检查是否所有订单都被分配
        all_assigned = len(set(sum([v["assigned_orders"] for v in temp_car_states.values()], [])))
        if all_assigned != self.n_orders:
            total_distance += 10000  # 未全部分配的严重惩罚
        
        return total_distance

    def _update_velocity_position(self, particle_idx: int):
        """更新粒子位置（离散问题特殊处理）"""
        current_particle = self.particles[particle_idx]
        pbest_particle = self.pbest[particle_idx]
        
        # 对每个订单的分配进行概率性更新
        for order_id in self.order_ids:
            r1, r2 = np.random.random(), np.random.random()
            # 认知和社会影响概率
            cognitive_prob = self.c1 * r1
            social_prob = self.c2 * r2
            
            # 按概率更新分配
            if cognitive_prob > 0.5:
                current_particle[order_id] = pbest_particle[order_id]
            if social_prob > 0.5 and self.gbest is not None:
                current_particle[order_id] = self.gbest[order_id]

    def optimize(self) -> Tuple[Dict[str, str], float]:
        """执行粒子群优化求解最优匹配"""
        self._initialize_particles()
        
        for _ in range(self.max_iter):
            # 计算适应度
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
    # 1. 生成订单和车辆数据
    orders = generate_taxi_orders(ORDER_NUM)
    cars = generate_netcar_locations(CAR_NUM)
    print(f"生成数据：{len(orders)}个订单，{len(cars)}辆车")
    
    # 2. K-means聚类分群（聚类数量可调整）
    cluster_matcher = TaxiCarClusterMatcher(n_clusters=4)
    cluster_matcher.cluster_taxi_orders(orders)
    cluster_matcher.cluster_netcar_locations(cars)
    cluster_matcher.match_cluster_centers()
    subgroups = cluster_matcher.split_into_subgroups()
    
    # 3. 每个子群分别执行PSO匹配
    all_results = {}
    for subgroup_id, (sub_orders, sub_cars) in subgroups.items():
        print(f"\n处理子群体 {subgroup_id}：{len(sub_orders)}个订单，{len(sub_cars)}辆车")
        
        if not sub_orders or not sub_cars:
            print("子群体订单或车辆为空，跳过匹配")
            continue
        
        # 执行PSO优化
        pso_matcher = PSOOrderMatcher(
            sub_orders, 
            sub_cars,
            w=0.7, 
            c1=1.2, 
            c2=1.2, 
            max_iter=100, 
            pop_size=50
        )
        best_matching, best_fitness = pso_matcher.optimize()
        
        all_results[subgroup_id] = {
            "matching": best_matching,
            "total_distance": best_fitness,
            "order_count": len(sub_orders),
            "car_count": len(sub_cars)
        }
        
        # 打印匹配结果
        print(f"最优匹配总距离：{best_fitness:.2f}公里")
        for order_id, car_id in best_matching.items():
            print(f"订单 {order_id} → 车辆 {car_id}")

if __name__ == "__main__":
    main()

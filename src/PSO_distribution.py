import random
import numpy as np
from Car_generate import NetCarLocation, cal_km_by_lon_lat, generate_netcar_locations
from Order_generate import TaxiOrder, generate_taxi_orders

# -------------------------- PSO参数配置 --------------------------
POP_SIZE = 50          # 粒子群规模（粒子数）
MAX_ITER = 30          # 最大迭代次数
W = 0.7                # 惯性权重（平衡全局/局部搜索）
C1 = 1.5               # 认知系数（粒子自身经验）
C2 = 1.5               # 社会系数（群体经验）
MAX_MATCH_DISTANCE = 8 # 最大匹配距离（km）

# -------------------------- 多目标适应度函数 --------------------------
def fitness(particle, orders, cars):
    """计算粒子的适应度值（越小越好）"""
    total_wait_time = 0.0   # 总等待时间
    total_empty_km = 0.0    # 总空驶里程
    match_count = 0         # 匹配成功订单数
    matched_cars = set()    # 已匹配的司机（避免重复）

    for order_idx, car_idx in enumerate(particle):
        order = orders[order_idx]
        # 跳过无效匹配（司机越界/已匹配/离线）
        if car_idx >= len(cars) or car_idx in matched_cars or not cars[car_idx].is_online:
            continue
        car = cars[car_idx]
        # 计算司机到订单起点的距离
        distance = cal_km_by_lon_lat(
            car.lon, car.lat,
            order.start_lon, order.start_lat
        )
        # 距离约束过滤
        if distance > MAX_MATCH_DISTANCE:
            continue
        # 计算等待时间（距离/30km/h × 拥堵系数1.5 → 转分钟）
        wait_time = (distance / 30) * 60 * 1.5
        total_wait_time += wait_time
        total_empty_km += distance
        match_count += 1
        matched_cars.add(car_idx)
    
    # 多目标加权计算适应度
    avg_wait = total_wait_time / max(match_count, 1)
    avg_empty = total_empty_km / max(match_count, 1)
    satisfy_rate = match_count / len(orders)
    alpha, beta, gamma = 0.5, 0.3, 0.2  # 动态权重
    return alpha * avg_wait + beta * avg_empty + gamma * (1 - satisfy_rate)

# -------------------------- PSO核心实现 --------------------------
def pso_order_matching(orders, cars):
    """PSO求解订单-司机最优匹配"""
    n_orders = len(orders)   # 订单数（粒子维度）
    n_cars = len(cars)       # 司机数（每个维度的取值范围）
    
    # 1. 初始化粒子群（每个粒子是n_orders维的数组，值为司机ID）
    particles = []
    for _ in range(POP_SIZE):
        particle = [random.randint(0, n_cars-1) for _ in range(n_orders)]
        particles.append({
            "position": np.array(particle),  # 当前位置（匹配方案）
            "velocity": np.zeros(n_orders),  # 当前速度
            "pbest": np.array(particle),     # 个体最优位置
            "pbest_fitness": float('inf'),   # 个体最优适应度
        })
    # 全局最优初始化
    gbest = particles[0]["position"].copy()
    gbest_fitness = float('inf')

    # 2. 迭代搜索最优解
    for iter in range(MAX_ITER):
        for particle in particles:
            # 计算当前适应度
            current_fitness = fitness(particle["position"], orders, cars)
            
            # 更新个体最优
            if current_fitness < particle["pbest_fitness"]:
                particle["pbest_fitness"] = current_fitness
                particle["pbest"] = particle["position"].copy()
            
            # 更新全局最优
            if current_fitness < gbest_fitness:
                gbest_fitness = current_fitness
                gbest = particle["position"].copy()
            
            # 更新速度和位置（PSO核心公式）
            r1, r2 = random.random(), random.random()
            particle["velocity"] = (
                W * particle["velocity"] +
                C1 * r1 * (particle["pbest"] - particle["position"]) +
                C2 * r2 * (gbest - particle["position"])
            )
            # 位置约束（司机ID为整数，且在有效范围内）
            particle["position"] = np.clip(
                np.round(particle["position"] + particle["velocity"]),
                0, n_cars-1
            ).astype(int)
    
    # 3. 解析最优匹配结果
    match_result = {}
    matched_cars = set()
    for order_idx, car_idx in enumerate(gbest):
        order = orders[order_idx]
        car = cars[car_idx]
        # 过滤无效匹配（确保司机未重复匹配、在线、距离合规）
        if car_idx not in matched_cars and car.is_online:
            distance = cal_km_by_lon_lat(car.lon, car.lat, order.start_lon, order.start_lat)
            if distance <= MAX_MATCH_DISTANCE:
                match_result[order.order_id] = car.car_id
                matched_cars.add(car_idx)
    
    return match_result, gbest_fitness

# -------------------------- 结合你的代码调用 --------------------------
if __name__ == "__main__":
    
    orders = generate_taxi_orders(10000)  # 生成100个订单
    cars = generate_netcar_locations(50) # 生成50辆网约车
    
    # 2. PSO求解最优匹配
    match_result, fitness_value = pso_order_matching(orders, cars)
    
    # 3. 输出结果
    print(f"✅ PSO匹配完成，匹配成功{len(match_result)}个订单，适应度值：{fitness_value:.2f}")
    for order_id, car_id in list(match_result.items())[:5]:
        print(f"  订单{order_id} → 车辆{car_id}")

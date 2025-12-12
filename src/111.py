import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Optional

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
STORAGE_DIR = "./pso_matching_results"  # 结果存储目录

# 可视化配置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 负号显示
sns.set_style("whitegrid")
COLORS = sns.color_palette("husl", n_colors=N_CLUSTERS)  # 子群配色

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

# -------------------------- 子群内PSO匹配核心类 --------------------------
class SubgroupPSOMatcher:
    """子群内订单-车辆匹配的PSO算法实现"""
    def __init__(self, subgroup_id: int, orders: List[TaxiOrder], cars: List[NetCarLocation]):
        self.subgroup_id = subgroup_id
        self.orders = orders
        self.cars = cars
        self.n_orders = len(orders)
        self.n_cars = len(cars)
        self.n_match = min(self.n_orders, self.n_cars)

        # 预处理：车辆到订单起点的距离矩阵
        self.distance_matrix = self._build_distance_matrix()

        # PSO参数
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
        """构建车辆-订单起点距离矩阵"""
        dist_mat = np.zeros((self.n_cars, self.n_orders))
        for car_idx, car in enumerate(self.cars):
            for order_idx, order in enumerate(self.orders):
                dist_mat[car_idx, order_idx] = calculate_haversine_distance(
                    car.lon, car.lat, order.start_lon, order.start_lat
                )
        return dist_mat

    def _init_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """初始化粒子群"""
        positions = np.zeros((self.n_particles, self.n_match), dtype=int)
        velocities = np.random.uniform(-self.v_max, self.v_max, (self.n_particles, self.n_match))

        for i in range(self.n_particles):
            if self.n_orders >= self.n_match:
                positions[i] = np.random.choice(self.n_orders, size=self.n_match, replace=False)
            else:
                pos = np.random.choice(self.n_orders, size=self.n_orders, replace=False)
                padding = -1 * np.ones(self.n_match - self.n_orders, dtype=int)
                positions[i] = np.concatenate([pos, padding])
        return positions, velocities
    
    def _calculate_fitness(self, position: np.ndarray) -> float:
        """优化目标：70%匹配数量 + 30%距离"""
        # 1. 匹配数量得分（越小越好）
        matched_count = sum(1 for o in position if o != -1)
        count_score = 1 / (matched_count + 1e-6)
    
        # 2. 总距离得分（归一化，越小越好）
        total_dist = 0.0
        for car_idx, order_idx in enumerate(position):
            if order_idx == -1:
                continue
            total_dist += self.distance_matrix[car_idx, order_idx]
        dist_score = total_dist / (self.n_match * np.max(self.distance_matrix) + 1e-6)
    
        # 3. 加权求和
        return 0.3 * dist_score + 0.7 * count_score

    def _update_velocity(self, vel: np.ndarray, pos: np.ndarray, p_best: np.ndarray, g_best: np.ndarray) -> np.ndarray:
        """更新粒子速度"""
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
            if new_pos[i] in new_pos[:i]:
                available = [o for o in range(self.n_orders) if o not in new_pos[:i]]
                new_pos[i] = available[0] if available else -1
        return new_pos

    def run(self) -> Tuple[Dict[int, Optional[TaxiOrder]], float]:
        """执行PSO并返回匹配结果"""
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
                print(f"子群{self.subgroup_id} PSO迭代{iter_idx+1}/{self.max_iter} | 最优适应度：{self.best_global_fitness:.2f}")

        # 生成匹配结果
        match_result = {}
        for car_idx in range(self.n_cars):
            if car_idx < len(self.best_global_position):
                order_idx = self.best_global_position[car_idx]
                match_result[car_idx] = self.orders[order_idx] if (order_idx != -1 and order_idx < self.n_orders) else None
            else:
                match_result[car_idx] = None

        return match_result, self.best_global_fitness

# -------------------------- 结果保存函数 --------------------------
def save_pso_results(subgroups: Dict[int, Tuple[List[TaxiOrder], List[NetCarLocation]]],
                     pso_results: Dict[int, Tuple[Dict[int, Optional[TaxiOrder]], float]]):
    """保存PSO匹配结果为JSON"""
    Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    save_path = f"{STORAGE_DIR}/pso_matching_results.json"

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

        # 转换为纯基础类型
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

# -------------------------- 可视化函数 --------------------------
def plot_cluster_distribution(subgroups: Dict[int, Tuple[List[TaxiOrder], List[NetCarLocation]]], save_dir: str):
    """绘制所有订单和车辆的聚类分布（不同子群不同颜色）"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for subgroup_id, (orders, cars) in subgroups.items():
        color = COLORS[subgroup_id % len(COLORS)]
        
        # 绘制订单起点
        order_lons = [o.start_lon for o in orders]
        order_lats = [o.start_lat for o in orders]
        ax.scatter(order_lons, order_lats, color=color, s=50, alpha=0.7, 
                   label=f'子群{subgroup_id}订单', marker='o', edgecolors='black', linewidth=0.5)
        
        # 绘制车辆位置
        car_lons = [c.lon for c in cars]
        car_lats = [c.lat for c in cars]
        ax.scatter(car_lons, car_lats, color=color, s=80, alpha=0.8, 
                   label=f'子群{subgroup_id}车辆', marker='^', edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('经度', fontsize=12)
    ax.set_ylabel('纬度', fontsize=12)
    ax.set_title('订单与车辆聚类分布', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 保存图片
    save_path = Path(save_dir) / "cluster_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"聚类分布图已保存至：{save_path}")

def plot_subgroup_matches(subgroups: Dict[int, Tuple[List[TaxiOrder], List[NetCarLocation]]],
                          pso_results: Dict[int, Tuple[Dict[int, Optional[TaxiOrder]], float]],
                          save_dir: str):
    """绘制每个子群的匹配详情（车辆-订单匹配连线）"""
    n_subgroups = len(subgroups)
    n_cols = min(3, n_subgroups)
    n_rows = (n_subgroups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_subgroups == 1:
        axes = [axes]
    elif n_rows > 1:
        axes = axes.flatten()
    
    # 遍历每个子群绘制
    for idx, (subgroup_id, (orders, cars)) in enumerate(subgroups.items()):
        ax = axes[idx] if n_subgroups > 1 else axes
        color = COLORS[subgroup_id % len(COLORS)]
        match_result, _ = pso_results[subgroup_id]
        
        # 绘制订单起点
        order_lons = [o.start_lon for o in orders]
        order_lats = [o.start_lat for o in orders]
        ax.scatter(order_lons, order_lats, color='red', s=60, alpha=0.8, 
                   label='订单起点', marker='o', edgecolors='black', linewidth=0.5)
        
        # 绘制车辆位置
        car_lons = [c.lon for c in cars]
        car_lats = [c.lat for c in cars]
        ax.scatter(car_lons, car_lats, color='blue', s=80, alpha=0.8, 
                   label='车辆位置', marker='^', edgecolors='black', linewidth=0.5)
        
        # 绘制匹配连线
        for car_idx, order in match_result.items():
            if order is None:
                continue
            car = cars[car_idx]
            ax.plot([car.lon, order.start_lon], [car.lat, order.start_lat], 
                    color=color, alpha=0.6, linewidth=1.5, linestyle='--')
        
        # 设置子图标题和标签
        matched_count = sum(1 for v in match_result.values() if v is not None)
        ax.set_title(f'子群{subgroup_id}匹配详情\n匹配数：{matched_count}/{len(orders)}订单 | {len(cars)}车辆', 
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('经度', fontsize=9)
        ax.set_ylabel('纬度', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    if n_subgroups < len(axes):
        for idx in range(n_subgroups, len(axes)):
            axes[idx].set_visible(False)
    
    plt.suptitle('各子群订单-车辆匹配详情', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图片
    save_path = Path(save_dir) / "subgroup_matches.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"子群匹配详情图已保存至：{save_path}")

def plot_statistics(pso_results: Dict[int, Tuple[Dict[int, Optional[TaxiOrder]], float]],
                    subgroups: Dict[int, Tuple[List[TaxiOrder], List[NetCarLocation]]],
                    save_dir: str):
    """绘制统计图表：各子群匹配数、总距离柱状图"""
    subgroup_ids = sorted(subgroups.keys())
    matched_counts = []
    total_dists = []
    
    for sg_id in subgroup_ids:
        match_result, total_dist = pso_results[sg_id]
        matched_counts.append(sum(1 for v in match_result.values() if v is not None))
        total_dists.append(round(total_dist, 2))
    
    # 创建双轴柱状图
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # 绘制匹配数柱状图（左轴）
    x = np.arange(len(subgroup_ids))
    width = 0.35
    bars1 = ax1.bar(x - width/2, matched_counts, width, label='匹配订单数', color='#2E86AB', alpha=0.8)
    ax1.set_xlabel('子群ID', fontsize=12)
    ax1.set_ylabel('匹配订单数', fontsize=12, color='#2E86AB')
    ax1.tick_params(axis='y', labelcolor='#2E86AB')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'子群{sg_id}' for sg_id in subgroup_ids])
    
    # 绘制总距离柱状图（右轴）
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, total_dists, width, label='总行驶距离(km)', color='#A23B72', alpha=0.8)
    ax2.set_ylabel('总行驶距离(km)', fontsize=12, color='#A23B72')
    ax2.tick_params(axis='y', labelcolor='#A23B72')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.set_title('各子群匹配数与总行驶距离统计', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    save_path = Path(save_dir) / "statistics.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"统计图表已保存至：{save_path}")

# -------------------------- 主执行逻辑 --------------------------
def main():
    """主函数：执行PSO匹配 + 结果保存 + 可视化"""
    # 1. 生成数据并获取聚类子群
    print("===== 1. 生成数据并获取子群体（聚类逻辑解耦） =====")
    taxi_orders = generate_taxi_orders(ORDER_NUM)
    car_locations = generate_netcar_locations(CAR_NUM)
    clustered_orders, order_kmeans, _ = cluster_taxi_orders(taxi_orders, N_CLUSTERS)
    clustered_cars, car_kmeans, _ = cluster_netcar_locations(car_locations, N_CLUSTERS)
    center_matches, _ = match_cluster_centers(order_kmeans.cluster_centers_, car_kmeans.cluster_centers_)
    subgroups = split_into_subgroups(clustered_orders, clustered_cars, center_matches)

    # 2. 子群内PSO匹配
    print("\n===== 2. 执行子群内PSO订单匹配 =====")
    pso_results = {}
    for subgroup_id, (orders, cars) in subgroups.items():
        print(f"\n处理子群{subgroup_id}：订单数={len(orders)}，车辆数={len(cars)}")
        if len(orders) == 0 or len(cars) == 0:
            pso_results[subgroup_id] = ({}, 0.0)
            print(f"子群{subgroup_id}无订单/车辆，跳过")
            continue

        matcher = SubgroupPSOMatcher(subgroup_id, orders, cars)
        match_result, total_dist = matcher.run()
        pso_results[subgroup_id] = (match_result, total_dist)

        matched_count = sum(1 for v in match_result.values() if v is not None)
        print(f"子群{subgroup_id}匹配完成：匹配{matched_count}单，总适应度{total_dist:.2f}")

    # 3. 保存结果
    save_pso_results(subgroups, pso_results)

    # 4. 生成可视化结果
    print("\n===== 3. 生成可视化结果 =====")
    plot_cluster_distribution(subgroups, STORAGE_DIR)
    plot_subgroup_matches(subgroups, pso_results, STORAGE_DIR)
    plot_statistics(pso_results, subgroups, STORAGE_DIR)

    # 5. 汇总结果
    print("\n===== 4. PSO匹配结果汇总 =====")
    total_matched = sum(sum(1 for v in res[0].values() if v is not None) for res in pso_results.values())
    total_distance = sum(res[1] for res in pso_results.values())
    print(f"全局总匹配订单数：{total_matched}/{ORDER_NUM}")
    print(f"全局总适应度：{total_distance:.2f}")
    if total_matched > 0:
        print(f"平均每单适应度：{total_distance/total_matched:.2f}")

if __name__ == "__main__":
    main()

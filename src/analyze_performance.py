import time
import matplotlib.pyplot as plt
import numpy as np
import dragon_map  


test_configs = {
    # 规模参数：采样点从 7 个增加到 15 个，范围扩大
    'ANT_COUNT': np.linspace(5, 150, 15, dtype=int).tolist(),
    'MAX_ACO_ITERATIONS': np.linspace(5, 100, 15, dtype=int).tolist(),
    
    # 权重参数：增加采样密度，观察细微波动
    'ALPHA': np.linspace(0.1, 10, 20).tolist(),
    'BETA': np.linspace(0.1, 10, 20).tolist(),
    
    # 动态参数：在关键区间 [0, 1] 内进行高密度采样
    'RHO': np.linspace(0.01, 0.99, 20).tolist(),
    
    # 增量参数：使用对数刻度，观察跨数量级的影响
    'Q': [10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    
    # 初始环境参数
    'INITIAL_PHEROMONE': np.linspace(0.1, 10, 15).tolist()
}


TRIALS_PER_VALUE = 5

def run_single_test(param_name, value, planner, cars, users, dests):

    original_val = getattr(dragon_map, param_name)

    setattr(dragon_map, param_name, value)
    
    start_t = time.time()

    result = planner.match_users_by_total_distance(cars, users, dests)
    end_t = time.time()
    
    total_dist = sum(info[5] for info in result.values() if info[0] is not None and info[5] != float('inf'))
    
    setattr(dragon_map, param_name, original_val)
    return (end_t - start_t), total_dist

def main_analysis():
    from dragon_map import generate_random_xy, GRID_COLS, GRID_ROWS, HybridPathPlanner
    
    dragon_map.COUNT_user = 6
    dragon_map.COUNT_car = 8
    dragon_map.COUNT_stop = 30
    
    stops = generate_random_xy(dragon_map.COUNT_stop, GRID_COLS, GRID_ROWS)
    users = generate_random_xy(dragon_map.COUNT_user, GRID_COLS, GRID_ROWS, avoid=stops)
    dests = {}
    avoid = stops + users
    for u in users:
        d = generate_random_xy(1, GRID_COLS, GRID_ROWS, avoid=avoid)[0]
        dests[u] = d
        avoid.append(d)
    cars = generate_random_xy(dragon_map.COUNT_car, GRID_COLS, GRID_ROWS, avoid=avoid)
    
    planner = HybridPathPlanner(stops)

    for param, values in test_configs.items():
        times = []
        distances = []
        print(f"正在分析参数: {param}...")
        
        for v in values:
            t, d = run_single_test(param, v, planner, cars, users, dests)
            times.append(t)
            distances.append(d)
        
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        color_time = 'tab:red'
        ax1.set_xlabel(f'Parameter Value: {param}')
        ax1.set_ylabel('Execution Time (s)', color=color_time)
        ax1.plot(values, times, 'o-', color=color_time, label='Time')
        ax1.tick_params(axis='y', labelcolor=color_time)
        
        ax2 = ax1.twinx()
        color_dist = 'tab:blue'
        ax2.set_ylabel('Total Best Distance (Grids)', color=color_dist)
        ax2.plot(values, distances, 's--', color=color_dist, label='Distance')
        ax2.tick_params(axis='y', labelcolor=color_dist)
        
        plt.title(f'Impact of {param} on Performance')
        fig.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"analysis_{param}.png")
        print(f"图像已保存: analysis_{param}.png")
        plt.close()

if __name__ == "__main__":
    main_analysis()
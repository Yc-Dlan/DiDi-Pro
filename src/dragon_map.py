import pygame
import random
import sys
import numpy as np
from enum import Enum
import heapq

# ===================== 自定义颜色分类 =====================
class Color(Enum):
    BLACK = (0, 0, 0)          # 黑色（背景）
    WHITE = (255, 255, 255)    # 白色（文字以及网格线）
    GRAY = (50, 50, 50)        # 灰色
    RED = (255, 0, 0)          # 红色（禁止区域）
    GREEN = (0, 255, 0)        # 绿色（用户）
    BLUE = (0, 0, 255)         # 蓝色（车辆）
    PURPLE = (128, 0, 128)     # 紫色（用户目标地址）
    ORANGE = (255, 165, 0)     # 橙色
    YELLOW = (255, 255, 0)     # 黄色（接人路径：车辆→用户）
    CYAN = (0, 255, 255)       # 青色（送人路径：用户→目标地址）
    PINK = (255, 192, 203)     # 粉色 

    @property
    def rgb(self):
        return self.value

# ===================== 全局配置 =====================
GRID_SIZE = 30          # 格子大小
GRID_ROWS = 30          # 网格行数
GRID_COLS = 30          # 网格列数
WINDOW_WIDTH = GRID_COLS * GRID_SIZE 
WINDOW_HEIGHT = GRID_ROWS * GRID_SIZE 

# 随机生成参数
COUNT_user = 20   # 用户数量
COUNT_car = 30   # 车辆数量
COUNT_stop = 200  # 禁止区域数量

# 混合算法(A*+蚁群算法)参数
ALPHA = 2             # 信息素重要程度因子
BETA = 3              # 启发函数重要程度因子
RHO = 0.1                # 信息素挥发因子
Q = 100                   # 信息素增量常数
MAX_ACO_ITERATIONS = 30  # 蚁群最大迭代次数
ANT_COUNT = 30           # 蚂蚁数量
INITIAL_PHEROMONE = 1.0  # 初始信息素浓度



#===================== 画图函数 =====================#
def generate_random_xy(count, max_x, max_y, avoid=None):
    """生成不重复的(x,y)坐标，支持避障"""
    if avoid is None:
        avoid = set()
    else:
        avoid = set(avoid)
    coords = []
    used = set()
    while len(coords) < count:
        x = random.randint(0, max_x - 1)
        y = random.randint(0, max_y - 1)
        if (x, y) not in used and (x, y) not in avoid:
            used.add((x, y))
            coords.append((x, y))
    return coords

def draw_grid(screen, grid_color: Color = Color.WHITE):
    """绘制网格线"""
    for x in range(0, WINDOW_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, grid_color.rgb, (x, 0), (x, WINDOW_HEIGHT), 1)
    for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, grid_color.rgb, (0, y), (WINDOW_WIDTH, y), 1)

def draw_single_block(screen, x, y, block_color, value="", text_color: Color = Color.WHITE):
    """绘制单个方块"""
    block_rgb = block_color.rgb if isinstance(block_color, Color) else block_color
    px = x * GRID_SIZE
    py = y * GRID_SIZE
    pygame.draw.rect(screen, block_rgb, (px, py, GRID_SIZE, GRID_SIZE))
    if value != "":
        font = pygame.font.SysFont(None, int(GRID_SIZE * 0.7))
        text = font.render(str(value), True, text_color.rgb)
        text_rect = text.get_rect(center=(px + GRID_SIZE//2, py + GRID_SIZE//2))
        screen.blit(text, text_rect)

def draw_path(screen, path, color=Color.YELLOW, width=3):
    """绘制路径"""
    if len(path) < 2:
        return
    pixel_path = [(x*GRID_SIZE + GRID_SIZE//2, y*GRID_SIZE + GRID_SIZE//2) for x,y in path]
    pygame.draw.lines(screen, color.rgb, False, pixel_path, width)

#*************************路径规划（A*+蚁群算法优化）**********************#
# ===================== 混合算法核心类 =====================
class HybridPathPlanner:
    def __init__(self, obstacles):
        self.obstacles = set(obstacles)  # 禁止区域（障碍）集合
        self.cols = GRID_COLS
        self.rows = GRID_ROWS
        self.pheromone_map = np.ones((self.cols, self.rows)) * INITIAL_PHEROMONE
        
    def heuristic(self, pos, end):
        """曼哈顿距离启发函数"""
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])
    
    def get_neighbors(self, x, y):
        """获取上下左右合法邻居（非障碍+在网格内）"""
        neighbors = []
        directions = [(0,1), (0,-1), (1,0), (-1,0)]  
        for dx, dy in directions:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < self.cols and 0 <= ny < self.rows and (nx, ny) not in self.obstacles:
                neighbors.append((nx, ny))
        return neighbors
    
    def a_star_with_pheromone(self, start, end, use_pheromone=True):
        """A*算法结合信息素启发"""
        if start == end:
            return [], 0
        if start in self.obstacles or end in self.obstacles:
            return [], float('inf')
        
        open_set = []
        heapq.heappush(open_set, (0, 0, start))
        came_from = {}
        g_score = {start: 0}
        counter = 0
        
        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            
            if current == end:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1], len(path) - 1
            
            for neighbor in self.get_neighbors(*current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    # 计算启发值
                    h = self.heuristic(neighbor, end)
                    
                    if use_pheromone:
                        # 结合信息素的启发函数
                        pheromone_value = self.pheromone_map[neighbor[0]][neighbor[1]]
                        f = tentative_g + h * (1.0 / (pheromone_value + 1e-6))
                    else:
                        f = tentative_g + h
                    
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))
        
        return [], float('inf')
    
    def aco_optimize_path(self, start, end, initial_path=None):
        """蚁群算法优化路径"""
        if not initial_path or len(initial_path) == 0:
            return [], float('inf')
        
        if len(initial_path) <= 2:
            return initial_path, len(initial_path) - 1
        
        best_path = initial_path
        best_length = len(initial_path) - 1
        
        local_pheromone = self.pheromone_map.copy()
        
        for (x, y) in initial_path:
            local_pheromone[x][y] += 5.0
        
        for iteration in range(MAX_ACO_ITERATIONS):
            ant_paths = []
            ant_lengths = []
            
            for ant in range(ANT_COUNT):
                current = start
                path = [current]
                visited = set([current])
                
                while current != end:
                    neighbors = self.get_neighbors(*current)
                    neighbors = [n for n in neighbors if n not in visited]
                    
                    if not neighbors:
                        break
                    
                    # 计算选择概率
                    probabilities = []
                    for neighbor in neighbors:
                        tau = local_pheromone[neighbor[0]][neighbor[1]] ** ALPHA
                        eta = (1.0 / (self.heuristic(neighbor, end) + 1e-6)) ** BETA
                        probabilities.append(tau * eta)
                    
                    # 轮盘赌选择
                    total = sum(probabilities)
                    if total == 0:
                        next_pos = random.choice(neighbors)
                    else:
                        rand_val = random.uniform(0, total)
                        cumulative = 0
                        for i, neighbor in enumerate(neighbors):
                            cumulative += probabilities[i]
                            if cumulative >= rand_val:
                                next_pos = neighbor
                                break
                        else:
                            next_pos = neighbors[-1]
                    
                    path.append(next_pos)
                    visited.add(next_pos)
                    current = next_pos
                
                if current == end:
                    length = len(path) - 1
                    ant_paths.append(path)
                    ant_lengths.append(length)
                    
                    if length < best_length:
                        best_length = length
                        best_path = path
            
            # 更新信息素
            local_pheromone *= (1 - RHO)
            
            # 增强最优路径的信息素
            if best_length > 0:
                pheromone_deposit = Q / best_length
                for (x, y) in best_path:
                    local_pheromone[x][y] += pheromone_deposit
        
        # 更新全局信息素地图
        self.pheromone_map = 0.7 * self.pheromone_map + 0.3 * local_pheromone
        
        return best_path, best_length
    
    def hybrid_find_path(self, start, end):
        """改进的混合算法"""
        # 阶段1：先用A*找到一条基础路径
        a_star_path, a_star_dist = self.a_star_with_pheromone(start, end, use_pheromone=False)
        
        if a_star_dist == float('inf'):
            return [], float('inf')
        
        # 阶段2：如果路径较短，直接返回A*结果
        if a_star_dist < 10:  # 阈值
            return a_star_path, a_star_dist
        
        # 阶段3：用蚁群算法优化较长路径
        aco_path, aco_dist = self.aco_optimize_path(start, end, a_star_path)
        
        # 选择更优
        if aco_dist < a_star_dist and len(aco_path) > 0:
            final_path = aco_path
            final_dist = aco_dist
        else:
            final_path = a_star_path
            final_dist = a_star_dist
        
        # 更新信息素
        if final_dist > 0:
            pheromone_deposit = Q / final_dist
            for (x, y) in final_path:
                # 路径上的每个点都增强
                self.pheromone_map[x][y] = (1 - RHO) * self.pheromone_map[x][y] + pheromone_deposit
            # 起点和终点额外增强
            self.pheromone_map[start[0]][start[1]] += pheromone_deposit * 2
            self.pheromone_map[end[0]][end[1]] += pheromone_deposit * 2
        
        return final_path, final_dist
  
    def match_users_by_total_distance(self, cars, users, user_dests):
        """
        以总距离最小化为目标，为所有用户匹配车辆
        采用贪心策略：按总距离从小到大排序，优先分配总距离最小的匹配
        """
        # 重置信息素地图
        #self.pheromone_map = np.ones((self.cols, self.rows)) * INITIAL_PHEROMONE
        
        matched = {}  # 用户 -> 匹配信息
        car_assignments = {}  # 车辆 -> 用户
        available_cars = set(cars)  # 可用的车辆集合
        
        # 为每个用户计算所有可能车辆的总距离
        all_possible_matches = []
        
        for user in users:
            dest = user_dests[user]
            for car in cars:
                if car in available_cars:
                    car_to_user_path, car_to_user_dist = self.hybrid_find_path(car, user)
                    user_to_dest_path, user_to_dest_dist = self.hybrid_find_path(user, dest)
                    
                    if car_to_user_dist < float('inf') and user_to_dest_dist < float('inf'):
                        total_distance = car_to_user_dist + user_to_dest_dist
                        all_possible_matches.append((total_distance, user, car))
        
        # 按总距离排序
        all_possible_matches.sort(key=lambda x: x[0])
        
        # 贪心分配
        matched_users = set()
        
        for total_distance, user, car in all_possible_matches:
            if user in matched_users or car not in available_cars:
                continue
                
            dest = user_dests[user]
            # 重新计算路径（确保使用正确的路径）
            car_to_user_path, car_to_user_dist = self.hybrid_find_path(car, user)
            user_to_dest_path, user_to_dest_dist = self.hybrid_find_path(user, dest)
            
            matched[user] = (car, car_to_user_path, car_to_user_dist, 
                           user_to_dest_path, user_to_dest_dist, total_distance)
            car_assignments[car] = user
            available_cars.remove(car)
            matched_users.add(user)
            
            if len(matched_users) == len(users) or not available_cars:
                break
        
        # 处理未匹配的用户
        for user in users:
            if user not in matched:
                matched[user] = (None, [], float('inf'), [], float('inf'), float('inf'))
        
        return matched

    def pure_a_star(self, start, end):
    #"""纯A*算法（不使用信息素）"""
     return self.a_star_with_pheromone(start, end, use_pheromone=False)

    def pure_aco(self, start, end):
        """纯蚁群算法（不依赖A*初始路径）"""
        if start == end:
            return [], 0
        if start in self.obstacles or end in self.obstacles:
            return [], float('inf')
        
        best_path = []
        best_length = float('inf')
        local_pheromone = np.ones((self.cols, self.rows)) * INITIAL_PHEROMONE
        
        for iteration in range(MAX_ACO_ITERATIONS):
            ant_paths = []
            ant_lengths = []
            
            for ant in range(ANT_COUNT):
                current = start
                path = [current]
                visited = set([current])
                
                while current != end:
                    neighbors = self.get_neighbors(*current)
                    neighbors = [n for n in neighbors if n not in visited]
                    
                    if not neighbors:
                        break  # 陷入死胡同
                    
                    # 计算选择概率
                    probabilities = []
                    for neighbor in neighbors:
                        tau = local_pheromone[neighbor[0]][neighbor[1]] ** ALPHA
                        eta = (1.0 / (self.heuristic(neighbor, end) + 1e-6)) ** BETA
                        probabilities.append(tau * eta)
                    
                    # 轮盘赌选择
                    total = sum(probabilities)
                    if total == 0:
                        next_pos = random.choice(neighbors)
                    else:
                        rand_val = random.uniform(0, total)
                        cumulative = 0
                        for i, neighbor in enumerate(neighbors):
                            cumulative += probabilities[i]
                            if cumulative >= rand_val:
                                next_pos = neighbor
                                break
                        else:
                            next_pos = neighbors[-1]
                    
                    path.append(next_pos)
                    visited.add(next_pos)
                    current = next_pos
                
                if current == end:
                    length = len(path) - 1
                    ant_paths.append(path)
                    ant_lengths.append(length)
                    
                    if length < best_length:
                        best_length = length
                        best_path = path
            
            # 更新信息素
            local_pheromone *= (1 - RHO)
            
            # 增强最优路径的信息素
            if best_length != float('inf'):
                pheromone_deposit = Q / best_length
                for (x, y) in best_path:
                    local_pheromone[x][y] += pheromone_deposit
        
        return best_path, best_length

    def match_users_pure_a_star(self, cars, users, user_dests):
        """使用纯A*算法进行匹配"""
        matched = {}
        car_assignments = {}
        available_cars = set(cars)
        all_possible_matches = []
        
        for user in users:
            dest = user_dests[user]
            for car in cars:
                if car in available_cars:
                    car_to_user_path, car_to_user_dist = self.pure_a_star(car, user)
                    user_to_dest_path, user_to_dest_dist = self.pure_a_star(user, dest)
                    
                    if car_to_user_dist < float('inf') and user_to_dest_dist < float('inf'):
                        total_distance = car_to_user_dist + user_to_dest_dist
                        all_possible_matches.append((total_distance, user, car))
        
        all_possible_matches.sort(key=lambda x: x[0])
        matched_users = set()
        
        for total_distance, user, car in all_possible_matches:
            if user in matched_users or car not in available_cars:
                continue
                
            dest = user_dests[user]
            car_to_user_path, car_to_user_dist = self.pure_a_star(car, user)
            user_to_dest_path, user_to_dest_dist = self.pure_a_star(user, dest)
            
            matched[user] = (car, car_to_user_path, car_to_user_dist, 
                        user_to_dest_path, user_to_dest_dist, total_distance)
            car_assignments[car] = user
            available_cars.remove(car)
            matched_users.add(user)
            
            if len(matched_users) == len(users) or not available_cars:
                break
        
        for user in users:
            if user not in matched:
                matched[user] = (None, [], float('inf'), [], float('inf'), float('inf'))
        
        return matched

    def match_users_pure_aco(self, cars, users, user_dests):
        """使用纯蚁群算法进行匹配"""
        self.pheromone_map = np.ones((self.cols, self.rows)) * INITIAL_PHEROMONE
        
        matched = {}
        car_assignments = {}
        available_cars = set(cars)
        all_possible_matches = []
        
        for user in users:
            dest = user_dests[user]
            for car in cars:
                if car in available_cars:
                    car_to_user_path, car_to_user_dist = self.pure_aco(car, user)
                    user_to_dest_path, user_to_dest_dist = self.pure_aco(user, dest)
                    
                    if car_to_user_dist < float('inf') and user_to_dest_dist < float('inf'):
                        total_distance = car_to_user_dist + user_to_dest_dist
                        all_possible_matches.append((total_distance, user, car))
        
        all_possible_matches.sort(key=lambda x: x[0])
        matched_users = set()
        
        for total_distance, user, car in all_possible_matches:
            if user in matched_users or car not in available_cars:
                continue
                
            dest = user_dests[user]
            car_to_user_path, car_to_user_dist = self.pure_aco(car, user)
            user_to_dest_path, user_to_dest_dist = self.pure_aco(user, dest)
            
            matched[user] = (car, car_to_user_path, car_to_user_dist, 
                        user_to_dest_path, user_to_dest_dist, total_distance)
            car_assignments[car] = user
            available_cars.remove(car)
            matched_users.add(user)
            
            if len(matched_users) == len(users) or not available_cars:
                break
        
        for user in users:
            if user not in matched:
                matched[user] = (None, [], float('inf'), [], float('inf'), float('inf'))
        
        return matched   
    



def print_matching_summary(match_result, user_dests):
    """打印详细的匹配摘要信息"""
    print("\n" + "="*80)
    print("匹配结果总结")
    print("="*80)
    total_distance = 0
    successful_matches = 0
    user_distances = []
    
    for idx, (user_pos, match_info) in enumerate(match_result.items(), 1):
        car_pos, car2user_path, car2user_dist, user2dest_path, user2dest_dist, total_dist = match_info
        dest_pos = user_dests[user_pos]
        
        print(f"\n用户{idx}:")
        print(f"  位置: {user_pos} -> 目的地: {dest_pos}")
        
        if car_pos:
            successful_matches += 1
            print(f"  匹配车辆: {car_pos}")
            print(f"  车辆到用户距离: {car2user_dist} 格")
            print(f"  用户到目的地距离: {user2dest_dist} 格")
            print(f"  总距离: {total_dist} 格")
            print(f"  路径详情:")
            print(f"    车辆→用户: {car2user_path[:3]}...{car2user_path[-3:] if len(car2user_path) > 6 else ''}" + 
                  f" (共{len(car2user_path)}个点)")
            print(f"    用户→目的地: {user2dest_path[:3]}...{user2dest_path[-3:] if len(user2dest_path) > 6 else ''}" + 
                  f" (共{len(user2dest_path)}个点)")
            
            total_distance += total_dist
            user_distances.append((idx, total_dist))
        else:
            print(f"  ❌ 无可用车辆匹配")
            user_distances.append((idx, float('inf')))
    
    print("\n" + "="*80)
    print("总体统计:")
    print(f"  成功匹配用户数: {successful_matches}/{len(match_result)}")
    print(f"  总路程距离: {total_distance} 格")
    
    if successful_matches > 0:
        avg_distance = total_distance / successful_matches
        print(f"  平均每单距离: {avg_distance:.2f} 格")

    return total_distance, successful_matches

#******************************主函数**************************************#
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("A*加蚁群算法")
    clock = pygame.time.Clock()
    # ***********************生成场景：障碍，用户，目的地，车辆*******************#
    def generate_scene():
        stop_place = generate_random_xy(COUNT_stop, GRID_COLS, GRID_ROWS)#障碍
        user_list = generate_random_xy(COUNT_user, GRID_COLS, GRID_ROWS, avoid=stop_place)
        user_dests = {}
        avoid_dests = stop_place + user_list
        #防止生成重复在同一个块
        for u in user_list:
            dest = generate_random_xy(1, GRID_COLS, GRID_ROWS, avoid=avoid_dests)[0]
            user_dests[u] = dest
            avoid_dests.append(dest)
        car_list = generate_random_xy(COUNT_car, GRID_COLS, GRID_ROWS, avoid=avoid_dests)
        
        print(f"生成 {len(stop_place)} 个障碍物, {len(user_list)} 个用户, {len(car_list)} 辆车")
        return stop_place, user_list, user_dests, car_list
    
    stop_place, user_list, user_dests, car_list = generate_scene()
    screen.fill(Color.BLACK.rgb)
    draw_grid(screen, Color.WHITE)
    # 1. 绘制禁止区域
    for (x, y) in stop_place:
        draw_single_block(screen, x, y, Color.RED, value="X") 
    # 2. 绘制用户（标注U+序号）
    for idx, (x, y) in enumerate(user_list):
        draw_single_block(screen, x, y, Color.GREEN, value=f"U{idx}")
    
    # 3. 绘制目的地
    for idx, (user_pos, dest_pos) in enumerate(user_dests.items()):
        dx, dy = dest_pos
        draw_single_block(screen, dx, dy, Color.PURPLE, value=f"D{idx}")
    
    # 4. 绘制车辆（标注C+序号，避免和目的地重复）
    for idx, (x, y) in enumerate(car_list):
        draw_single_block(screen, x, y, Color.BLUE, value=f"C{idx}")

    #************************路径规划********************#
   # 初始化混合路径规划器
    planner = HybridPathPlanner(stop_place)
    
    # 进行三种算法的匹配
    print("\n===== 开始三种算法对比测试 =====")
    
    print("\n1. 混合算法（A*+蚁群）路径规划和匹配...")
    hybrid_result = planner.match_users_by_total_distance(car_list, user_list, user_dests)
    hybrid_total, hybrid_success = print_matching_summary(hybrid_result, user_dests)
    
    print("\n2. 纯A*算法路径规划和匹配...")
    a_star_result = planner.match_users_pure_a_star(car_list, user_list, user_dests)
    a_star_total, a_star_success = print_matching_summary(a_star_result, user_dests)
    
    print("\n3. 纯蚁群算法路径规划和匹配...")
    aco_result = planner.match_users_pure_aco(car_list, user_list, user_dests)
    aco_total, aco_success = print_matching_summary(aco_result, user_dests)
    
    # 打印算法对比总结
    print("\n" + "="*80)
    print("三种算法效果对比")
    print("="*80)

    print(f"| 算法类型       | 总距离（格） | 成功匹配数 | 平均每单距离（格） |")
    print(f"|----------------|--------------|------------|--------------------|")
    
    hybrid_avg = hybrid_total / hybrid_success if hybrid_success > 0 else 0
    print(f"| 混合算法       | {hybrid_total:12d} | {hybrid_success:10d} | {hybrid_avg:18.2f} |")
    
    a_star_avg = a_star_total / a_star_success if a_star_success > 0 else 0
    print(f"| 纯A*算法       | {a_star_total:12d} | {a_star_success:10d} | {a_star_avg:18.2f} |")
    
    aco_avg = aco_total / aco_success if aco_success > 0 else 0
    print(f"| 纯蚁群算法     | {aco_total:12d} | {aco_success:10d} | {aco_avg:18.2f} |")

    print("="*80)


    # 打印详细的匹配信息
    total_distance, successful_matches = print_matching_summary(hybrid_result, user_dests)
    # **********************主循环控制****************#
    running = True
    while (running==True):

        # 路径
        for idx, (user_pos, match_info) in enumerate(hybrid_result.items(), 1):
            car_pos, car2user_path, _, user2dest_path, _, total_dist = match_info
            dest_pos = user_dests[user_pos]
            
            # 绘制匹配路径
            if car_pos:
                # 绘制车到用户路径（黄色）
                draw_path(screen, car2user_path, Color.YELLOW, width=4)
                # 绘制用户到目的地路径（青色）
                draw_path(screen, user2dest_path, Color.CYAN, width=3)
        # 退出
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        #延时刷新       
        pygame.display.flip()
        clock.tick(20)


    #结束，退出
    pygame.quit()
    sys.exit()    
    


if __name__ == "__main__":
    main()
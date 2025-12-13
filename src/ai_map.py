import pygame
import random
import tkinter as tk
from enum import Enum
import math
import numpy as np

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
    PINK = (255, 192, 203)     # 粉色（扩展）

    @property
    def rgb(self):
        return self.value

# ===================== 全局配置 =====================
GRID_SIZE = 50          # 格子大小
GRID_ROWS = 20          # 网格行数
GRID_COLS = 20          # 网格列数
WINDOW_WIDTH = GRID_COLS * GRID_SIZE
WINDOW_HEIGHT = GRID_ROWS * GRID_SIZE

# 随机生成参数
COUNT_user = 15  # 用户数量
COUNT_car = 20   # 车辆数量
COUNT_stop = 50  # 禁止区域数量

# 蚁群算法参数（单用户场景下仅用于验证最短路径）
ANT_COUNT = 10           # 蚂蚁数量
MAX_ITERATIONS = 20      # 最大迭代次数
ALPHA = 1.0              # 信息素重要程度因子
BETA = 2.0               # 启发函数重要程度因子
RHO = 0.5                # 信息素挥发因子
Q = 100                  # 信息素增量常数

# ===================== A*路径规划 + 单用户最短匹配核心类 =====================
class AStarPlanner:
    def __init__(self, obstacles):
        self.obstacles = set(obstacles)  # 禁止区域（障碍）集合
        self.cols = GRID_COLS
        self.rows = GRID_ROWS

    def heuristic(self, pos, end):
        """曼哈顿距离启发函数"""
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def get_neighbors(self, x, y):
        """获取上下左右合法邻居（非障碍+在网格内）"""
        neighbors = []
        directions = [(0,1), (0,-1), (1,0), (-1,0)]  # 四方向
        for dx, dy in directions:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < self.cols and 0 <= ny < self.rows and (nx, ny) not in self.obstacles:
                neighbors.append((nx, ny))
        return neighbors

    def a_star_path(self, start, end):
        """A*算法计算避障路径，返回路径列表和路径长度"""
        if start == end:
            return [], 0
        if start in self.obstacles or end in self.obstacles:
            return [], float('inf')
        if (start[0] <0 or start[0]>=self.cols or start[1]<0 or start[1]>=self.rows) or \
           (end[0] <0 or end[0]>=self.cols or end[1]<0 or end[1]>=self.rows):
            return [], float('inf')

        open_set = [(self.heuristic(start, end), start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, end)}

        while open_set:
            open_set.sort()
            current = open_set.pop(0)[1]

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path = path[::-1]
                return path, len(path) - 1  # 返回路径 + 路径长度（格子数）

            for neighbor in self.get_neighbors(*current):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, end)
                    if (f_score[neighbor], neighbor) not in open_set:
                        open_set.append((f_score[neighbor], neighbor))

        return [], float('inf')  # 无路径

    def find_nearest_car_for_user(self, user_pos, dest_pos, cars, use_aco=False):
        """
        为单个用户找最近的车辆（核心：单用户独立计算）
        :param user_pos: 当前用户坐标
        :param dest_pos: 当前用户目标地址
        :param cars: 所有车辆坐标列表
        :param use_aco: 是否用蚁群算法验证最短路径
        :return: 最优车辆、车到用户路径+距离、用户到目的地路径+距离
        """
        min_car_to_user_dist = float('inf')
        best_car = None
        best_car_to_user_path = []
        best_user_to_dest_path = []
        best_user_to_dest_dist = float('inf')

        # 遍历所有车辆，找离当前用户最近的
        for car in cars:
            # 计算车辆到用户的路径和距离
            car_to_user_path, car_to_user_dist = self.a_star_path(car, user_pos)
            
            # 仅保留距离更短的车辆
            if car_to_user_dist < min_car_to_user_dist:
                min_car_to_user_dist = car_to_user_dist
                best_car = car
                best_car_to_user_path = car_to_user_path
                # 计算该用户到目的地的路径和距离（固定）
                best_user_to_dest_path, best_user_to_dest_dist = self.a_star_path(user_pos, dest_pos)

        # 蚁群算法验证（单用户场景下仅校验最短路径）
        if use_aco and best_car is not None:
            aco_dist = self.aco_single_path(best_car, user_pos)
            if aco_dist < min_car_to_user_dist and aco_dist != float('inf'):
                min_car_to_user_dist = aco_dist
                # 重新获取蚁群算法优化后的路径
                best_car_to_user_path, _ = self.a_star_path(best_car, user_pos)

        return best_car, best_car_to_user_path, min_car_to_user_dist, best_user_to_dest_path, best_user_to_dest_dist

    def aco_single_path(self, start, end):
        """蚁群算法计算单起点-单终点的最短路径（适配单用户场景）"""
        # 初始化信息素矩阵
        pheromone = np.ones((self.cols, self.rows)) * 0.1
        best_dist = float('inf')

        for _ in range(MAX_ITERATIONS):
            ant_paths = []
            ant_dists = []

            # 每只蚂蚁走一次路径
            for _ in range(ANT_COUNT):
                current = start
                path = [current]
                dist = 0
                visited = set([current])

                while current != end and dist < self.cols * self.rows:  # 防止死循环
                    neighbors = self.get_neighbors(*current)
                    neighbors = [n for n in neighbors if n not in visited]
                    
                    if not neighbors:
                        break  # 无可用邻居，终止

                    # 计算邻居概率
                    probs = []
                    for n in neighbors:
                        tau = pheromone[n[0]][n[1]]
                        eta = 1.0 / (self.heuristic(current, end) + 1e-6)
                        probs.append(tau ** ALPHA * eta ** BETA)
                    
                    # 轮盘赌选择
                    total = sum(probs)
                    if total == 0:
                        next_pos = random.choice(neighbors)
                    else:
                        r = random.uniform(0, total)
                        cumulative = 0
                        next_pos = None
                        for n, p in zip(neighbors, probs):
                            cumulative += p
                            if cumulative >= r:
                                next_pos = n
                                break

                    path.append(next_pos)
                    visited.add(next_pos)
                    dist += 1
                    current = next_pos

                # 记录有效路径
                if current == end:
                    ant_paths.append(path)
                    ant_dists.append(dist)
                    if dist < best_dist:
                        best_dist = dist

            # 信息素挥发+更新
            pheromone *= (1 - RHO)
            for path, dist in zip(ant_paths, ant_dists):
                for (x, y) in path:
                    pheromone[x][y] += Q / dist

        return best_dist if best_dist != float('inf') else float('inf')

    def match_each_user_individually(self, cars, users, user_dests, use_aco=False):
        """
        为每个用户单独匹配最近车辆（核心修改点）
        :return: 匹配字典 {用户坐标: (车辆坐标, 车到用户路径, 车到用户距离, 用户到目的地路径, 用户到目的地距离)}
        """
        matched = {}
        for user in users:
            dest = user_dests[user]
            # 为当前用户找最近的车辆
            best_car, car2user_path, car2user_dist, user2dest_path, user2dest_dist = self.find_nearest_car_for_user(
                user, dest, cars, use_aco
            )
            if best_car:
                matched[user] = (best_car, car2user_path, car2user_dist, user2dest_path, user2dest_dist)
            else:
                matched[user] = (None, [], float('inf'), [], float('inf'))  # 无可用车辆
        return matched

# ===================== 辅助函数 =====================
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

def draw_path(screen, path, color=Color.YELLOW):
    """绘制路径"""
    if len(path) < 2:
        return
    pixel_path = [(x*GRID_SIZE + GRID_SIZE//2, y*GRID_SIZE + GRID_SIZE//2) for x,y in path]
    pygame.draw.lines(screen, color.rgb, False, pixel_path, 3)

# ===================== 主函数 =====================
def main():
    # 初始化Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("单用户最短距离匹配 - 打印车到用户/用户到目的地距离")
    clock = pygame.time.Clock()

    # 生成坐标（禁止区→用户→用户目标→车辆，逐级避障）
    stop_place = generate_random_xy(COUNT_stop, GRID_COLS, GRID_ROWS)
    user_list = generate_random_xy(COUNT_user, GRID_COLS, GRID_ROWS, avoid=stop_place)
    user_dests = {}
    avoid_dests = stop_place + user_list
    for u in user_list:
        dest = generate_random_xy(1, GRID_COLS, GRID_ROWS, avoid=avoid_dests)[0]
        user_dests[u] = dest
        avoid_dests.append(dest)
    car_list = generate_random_xy(COUNT_car, GRID_COLS, GRID_ROWS, avoid=avoid_dests)

    # 初始化规划器，为每个用户单独匹配最近车辆
    planner = AStarPlanner(stop_place)
    use_aco = False  # 可切换是否用蚁群算法验证
    match_result = planner.match_each_user_individually(car_list, user_list, user_dests, use_aco)

    # ========== 核心：打印每个用户的详细距离 ==========
    print("========== 每个用户的最短距离匹配结果 ==========")
    for idx, (user_pos, match_info) in enumerate(match_result.items(), 1):
        car_pos, car2user_path, car2user_dist, user2dest_path, user2dest_dist = match_info
        print(f"\n用户{idx} 坐标：{user_pos}")
        if car_pos:
            print(f"  匹配车辆坐标：{car_pos}")
            print(f"  车辆到用户的距离：{car2user_dist} 格")
            print(f"  用户到目的地的距离：{user2dest_dist} 格")
        else:
            print(f"  无可用车辆匹配")
    print("==============================================")

    # Pygame主循环
    running = True
    while running:
        screen.fill(Color.BLACK.rgb)
        draw_grid(screen)

        # 绘制禁止区域
        for (x, y) in stop_place:
            draw_single_block(screen, x, y, Color.RED, value=f"X")

        # 绘制用户、目标地址、匹配车辆及路径
        for idx, (user_pos, match_info) in enumerate(match_result.items(), 1):
            car_pos, car2user_path, _, user2dest_path, _ = match_info
            ux, uy = user_pos
            dest_pos = user_dests[user_pos]
            dx, dy = dest_pos

            # 绘制用户
            draw_single_block(screen, ux, uy, Color.GREEN, value=f"U{idx}")
            # 绘制用户目标地址
            draw_single_block(screen, dx, dy, Color.PURPLE, value=f"D{idx}")
            # 绘制匹配车辆（若有）
            if car_pos:
                cx, cy = car_pos
                draw_single_block(screen, cx, cy, Color.BLUE, value=f"C{idx}")
                # 绘制车到用户路径（黄色）
                draw_path(screen, car2user_path, Color.YELLOW)
                # 绘制用户到目的地路径（青色）
                draw_path(screen, user2dest_path, Color.CYAN)

        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 空格键：重新生成场景+匹配
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                stop_place = generate_random_xy(COUNT_stop, GRID_COLS, GRID_ROWS)
                user_list = generate_random_xy(COUNT_user, GRID_COLS, GRID_ROWS, avoid=stop_place)
                user_dests = {}
                avoid_dests = stop_place + user_list
                for u in user_list:
                    dest = generate_random_xy(1, GRID_COLS, GRID_ROWS, avoid=avoid_dests)[0]
                    user_dests[u] = dest
                    avoid_dests.append(dest)
                car_list = generate_random_xy(COUNT_car, GRID_COLS, GRID_ROWS, avoid=avoid_dests)
                match_result = planner.match_each_user_individually(car_list, user_list, user_dests, use_aco)
                # 重新打印结果
                print("\n========== 重新匹配结果 ==========")
                for idx, (user_pos, match_info) in enumerate(match_result.items(), 1):
                    car_pos, car2user_path, car2user_dist, user2dest_path, user2dest_dist = match_info
                    print(f"\n用户{idx} 坐标：{user_pos}")
                    if car_pos:
                        print(f"  匹配车辆坐标：{car_pos}")
                        print(f"  车辆到用户的距离：{car2user_dist} 格")
                        print(f"  用户到目的地的距离：{user2dest_dist} 格")
                    else:
                        print(f"  无可用车辆匹配")
            # A键：切换是否使用蚁群算法验证
            if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                use_aco = not use_aco
                match_result = planner.match_each_user_individually(car_list, user_list, user_dests, use_aco)
                print(f"\n已切换蚁群算法验证：{'开启' if use_aco else '关闭'}，重新匹配完成")
                # 重新打印结果
                print("========== 切换算法后匹配结果 ==========")
                for idx, (user_pos, match_info) in enumerate(match_result.items(), 1):
                    car_pos, car2user_path, car2user_dist, user2dest_path, user2dest_dist = match_info
                    print(f"\n用户{idx} 坐标：{user_pos}")
                    if car_pos:
                        print(f"  匹配车辆坐标：{car_pos}")
                        print(f"  车辆到用户的距离：{car2user_dist} 格")
                        print(f"  用户到目的地的距离：{user2dest_dist} 格")
                    else:
                        print(f"  无可用车辆匹配")

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
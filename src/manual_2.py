import pygame
import random
import tkinter as tk
from enum import Enum

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

# ===================== 全局配置（先定义，避免顺序问题） =====================
GRID_SIZE = 50          # 缩小格子，避免窗口过大（可选）
GRID_ROWS = 20           # 网格行数（y的最大值：GRID_ROWS-1）
GRID_COLS = 20           # 网格列数（x的最大值：GRID_COLS-1）
WINDOW_WIDTH = GRID_COLS * GRID_SIZE
WINDOW_HEIGHT = GRID_ROWS * GRID_SIZE

# 随机生成参数（提前定义）
COUNT_user = 15  # 用户数量
COUNT_car = 20   # 车辆数量
COUNT_stop = 30  # 禁止数量

# ===================== A*路径规划 + 静态避障核心类 =====================
class AStarPlanner:
    def __init__(self, obstacles):
        self.obstacles = set(obstacles)  # 禁止区域（障碍）集合
        self.cols = GRID_COLS
        self.rows = GRID_ROWS

    def heuristic(self, pos, end):
        """曼哈顿距离启发函数（网格最优）"""
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
        """
        A*算法计算避障路径
        :param start: 起点(x,y)
        :param end: 终点(x,y)
        :return: 路径列表[(x1,y1), (x2,y2), ...]，空列表表示无路径
        """
        # 边界/障碍校验
        if start == end:
            return []
        if start in self.obstacles or end in self.obstacles:
            return []
        if (start[0] <0 or start[0]>=self.cols or start[1]<0 or start[1]>=self.rows) or \
           (end[0] <0 or end[0]>=self.cols or end[1]<0 or end[1]>=self.rows):
            return []

        # A*核心变量
        open_set = [(self.heuristic(start, end), start)]
        came_from = {}
        g_score = {start: 0}  # 起点到当前点代价
        f_score = {start: self.heuristic(start, end)}  # g + 启发值

        while open_set:
            # 取f_score最小的节点
            open_set.sort()
            current = open_set.pop(0)[1]

            # 到达终点，回溯路径
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # 反转路径（起点→终点）

            # 遍历邻居
            for neighbor in self.get_neighbors(*current):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, end)
                    if (f_score[neighbor], neighbor) not in open_set:
                        open_set.append((f_score[neighbor], neighbor))

        return []  # 无路径

    def match_car_user_dest(self, cars, users, user_dests):
        """
        多车辆-多用户匹配：为每个车辆分配路径最短的未匹配用户（含目标地址）
        :param cars: 车辆列表
        :param users: 用户列表
        :param user_dests: 用户目标地址字典 {用户坐标: 目标坐标}
        :return: 匹配字典 {车辆坐标: (用户坐标, 目标坐标, 接人路径, 送人路径)}
        """
        matched = {}
        unused_users = users.copy()

        for car in cars:
            if not unused_users:
                break
            
            min_total_len = float('inf')
            best_user = None
            best_dest = None
            best_pickup_path = []  # 接人路径：车辆→用户
            best_dropoff_path = [] # 送人路径：用户→目标

            # 计算当前车辆到所有未匹配用户的完整行程路径
            for user in unused_users:
                dest = user_dests[user]
                # 计算接人路径
                pickup_path = self.a_star_path(car, user)
                pickup_len = len(pickup_path) if pickup_path else float('inf')
                # 计算送人路径
                dropoff_path = self.a_star_path(user, dest)
                dropoff_len = len(dropoff_path) if dropoff_path else float('inf')
                # 总路径长度
                total_len = pickup_len + dropoff_len

                if total_len < min_total_len:
                    min_total_len = total_len
                    best_user = user
                    best_dest = dest
                    best_pickup_path = pickup_path
                    best_dropoff_path = dropoff_path

            # 记录匹配结果
            if best_user and best_dest:
                matched[car] = (best_user, best_dest, best_pickup_path, best_dropoff_path)
                unused_users.remove(best_user)

        return matched

# ===================== 核心函数（提前定义） =====================
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
    """绘制网格线（传入screen，避免作用域问题）"""
    for x in range(0, WINDOW_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, grid_color.rgb, (x, 0), (x, WINDOW_HEIGHT), 1)
    for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, grid_color.rgb, (0, y), (WINDOW_WIDTH, y), 1)

def draw_single_block(screen, x, y, block_color, value="", text_color: Color = Color.WHITE):
    """绘制单个方块"""
    # 处理颜色
    block_rgb = block_color.rgb if isinstance(block_color, Color) else block_color
    # 计算像素坐标
    px = x * GRID_SIZE
    py = y * GRID_SIZE
    # 画方块
    pygame.draw.rect(screen, block_rgb, (px, py, GRID_SIZE, GRID_SIZE))
    # 画数值（可选）
    if value != "":
        font = pygame.font.SysFont(None, int(GRID_SIZE * 0.7))
        text = font.render(str(value), True, text_color.rgb)
        text_rect = text.get_rect(center=(px + GRID_SIZE//2, py + GRID_SIZE//2))
        screen.blit(text, text_rect)

def draw_path(screen, path, color=Color.YELLOW):
    """绘制路径（网格坐标转像素坐标，中心点连线）"""
    if len(path) < 2:
        return
    pixel_path = [(x*GRID_SIZE + GRID_SIZE//2, y*GRID_SIZE + GRID_SIZE//2) for x,y in path]
    pygame.draw.lines(screen, color.rgb, False, pixel_path, 3)

# ===================== 主函数（核心逻辑） =====================
def main():
    # 1. 初始化Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("DIDIPRO - 一车一人一目的地 + 静态避障")
    clock = pygame.time.Clock()

    # 2. 生成坐标（核心：先生成禁止区域，再生成用户/车辆/目标地址避开禁止区域）
    stop_place = generate_random_xy(COUNT_stop, GRID_COLS, GRID_ROWS)  # 禁止区域
    user = generate_random_xy(COUNT_user, GRID_COLS, GRID_ROWS, avoid=stop_place)  # 用户避开禁止区
    # 为每个用户生成唯一目标地址（避开禁止区+用户+车辆）
    user_dests = {}
    avoid_dests = stop_place + user
    for u in user:
        dest = generate_random_xy(1, GRID_COLS, GRID_ROWS, avoid=avoid_dests)[0]
        user_dests[u] = dest
        avoid_dests.append(dest)  # 目标地址不重复
    # 生成车辆（避开禁止区+用户+目标地址）
    car = generate_random_xy(COUNT_car, GRID_COLS, GRID_ROWS, avoid=avoid_dests)

    # 3. 初始化A*规划器，执行多车辆-用户-目的地匹配
    planner = AStarPlanner(stop_place)
    match_result = planner.match_car_user_dest(car, user, user_dests)
        # 3. 初始化A*规划器，执行多车辆-用户-目的地匹配
    planner = AStarPlanner(stop_place)
    match_result = planner.match_car_user_dest(car, user, user_dests)

    # ========== 打印匹配明细 + 全局总距离 ==========
    print("========== 最终匹配明细 ==========")
    global_total = 0                                         # 新增：全局总行程
    for idx, (car_pos, (user_pos, dest_pos, pickup_path, dropoff_path)) in enumerate(match_result.items(), 1):
        pickup_len = len(pickup_path) - 1 if pickup_path else 0
        dropoff_len = len(dropoff_path) - 1 if dropoff_path else 0
        total_len = pickup_len + dropoff_len
        global_total += total_len                            # 累加
        print(f"C{idx}{car_pos}  →  U{user}{user_pos}  →  D{idx}{dest_pos}")
        print(f"  接人路径长度：{pickup_len}")
        print(f"  送人路径长度：{dropoff_len}")
        print(f"  总行程长度：{total_len}")
    print("=================================")
    print(f"全局总行程（所有车辆路程之和）：{global_total}")

    # 4. Pygame主循环（绘制逻辑放在这里）
    running = True
    while running:
        # 填充背景
        screen.fill(Color.BLACK.rgb)
        
        # 绘制网格
        draw_grid(screen)
        
        # 1. 绘制禁止区域（红色方块）
        for (x, y) in stop_place:
            draw_single_block(screen, x, y, Color.RED, value=f"X")
        
        # 2. 绘制用户目标地址（紫色方块）
        for idx, (u, dest) in enumerate(user_dests.items()):
            dx, dy = dest
            draw_single_block(screen, dx, dy, Color.PURPLE, value=f"D{idx+1}")
        
        # 3. 绘制用户（绿色方块）
        for idx, (x, y) in enumerate(user):
            draw_single_block(screen, x, y, Color.GREEN, value=f"U{idx+1}")
        
        # 4. 绘制车辆（蓝色方块）+ 两段路径
        for idx, (car_pos, (user_pos, dest_pos, pickup_path, dropoff_path)) in enumerate(match_result.items()):
            car_x, car_y = car_pos
            # 绘制车辆
            draw_single_block(screen, car_x, car_y, Color.BLUE, value=f"C{idx+1}")
            # 绘制接人路径（黄色：车辆→用户）
            draw_path(screen, pickup_path, Color.YELLOW)
            # 绘制送人路径（青色：用户→目标地址）
            draw_path(screen, dropoff_path, Color.CYAN)

        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 按空格键重新生成所有坐标并重新匹配
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                stop_place = generate_random_xy(COUNT_stop, GRID_COLS, GRID_ROWS)
                user = generate_random_xy(COUNT_user, GRID_COLS, GRID_ROWS, avoid=stop_place)
                # 重新生成用户目标地址
                user_dests = {}
                avoid_dests = stop_place + user
                for u in user:
                    dest = generate_random_xy(1, GRID_COLS, GRID_ROWS, avoid=avoid_dests)[0]
                    user_dests[u] = dest
                    avoid_dests.append(dest)
                car = generate_random_xy(COUNT_car, GRID_COLS, GRID_ROWS, avoid=avoid_dests)
                # 重新匹配
                planner = AStarPlanner(stop_place)
                match_result = planner.match_car_user_dest(car, user, user_dests)
                print(f"重新匹配完成：{len(match_result)} 辆车匹配到用户（含目的地）")

        # 更新屏幕
        pygame.display.flip()
        clock.tick(30)

    # 退出Pygame
    pygame.quit()

# ===================== 执行入口 =====================
if __name__ == "__main__":
    main()  # 调用主函数，统一执行逻辑
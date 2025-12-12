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
    PURPLE = (128, 0, 128)     # 紫色
    ORANGE = (255, 165, 0)     # 橙色
    YELLOW = (255, 255, 0)     # 黄色（扩展）
    CYAN = (0, 255, 255)       # 青色（扩展）
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
COUNT_car = 15   # 车辆数量
COUNT_stop = 30  # 禁止数量
# ===================== 核心函数（提前定义） =====================
def generate_random_xy(count, max_x, max_y):
    """生成不重复的(x,y)坐标：x=列，y=行"""
    coords = []
    used = set()
    while len(coords) < count:
        x = random.randint(0, max_x - 1)  # 修正：x范围0~max_x-1（避免越界）
        y = random.randint(0, max_y - 1)  # 修正：y范围0~max_y-1
        if (x, y) not in used:
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

# ===================== 主函数（核心逻辑） =====================
def main():
    # 1. 初始化Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("DIDIPRO")
    clock = pygame.time.Clock()

    # 2. 生成坐标（核心：在Pygame初始化后、主循环前生成）
    user = generate_random_xy(COUNT_user, GRID_COLS, GRID_ROWS)  # x=列，y=行
    car = generate_random_xy(COUNT_car, GRID_COLS, GRID_ROWS)            # 修正坐标顺序
    stop_place=generate_random_xy(COUNT_stop, GRID_COLS, GRID_ROWS)            # 修正坐标顺序
    # 3. Pygame主循环（绘制逻辑放在这里）
    running = True
    while running:
        # 填充背景
        screen.fill(Color.BLACK.rgb)
        
        # 绘制网格
        draw_grid(screen)
        
        # 绘制用户（绿色方块）
        for idx, (x, y) in enumerate(user):
            draw_single_block(screen, x, y, Color.GREEN, value=f"U{idx+1}")
        
        # 绘制车辆（蓝色方块）
        for idx, (x, y) in enumerate(car):
            draw_single_block(screen, x, y, Color.BLUE, value=f"C{idx+1}")

          # 绘制禁止区域（红色方块）
        for idx, (x, y) in enumerate(stop_place):
            draw_single_block(screen, x, y, Color.RED, value=f"C{idx+1}")

        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 更新屏幕
        pygame.display.flip()
        clock.tick(30)

    # 退出Pygame
    pygame.quit()

# ===================== 执行入口 =====================
if __name__ == "__main__":
    main()  # 调用主函数，统一执行逻辑
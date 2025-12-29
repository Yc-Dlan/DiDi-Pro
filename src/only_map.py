import pygame
import random
import sys
from enum import Enum

# ===================== 颜色与配置 =====================
class Color(Enum):
    BLACK = (0, 0, 0)          # 背景
    WHITE = (255, 255, 255)    # 网格线
    RED = (255, 0, 0)          # 障碍物
    GREEN = (0, 255, 0)        # 用户
    BLUE = (0, 0, 255)         # 车辆
    PURPLE = (128, 0, 128)     # 目的地

GRID_SIZE = 30          
GRID_ROWS = 30          
GRID_COLS = 30          
WINDOW_WIDTH = GRID_COLS * GRID_SIZE 
WINDOW_HEIGHT = GRID_ROWS * GRID_SIZE 

# 数量配置
COUNT_USER = 20   
COUNT_CAR = 22   
COUNT_STOP = 20  


def generate_random_xy(count, max_x, max_y, avoid=None):
    if avoid is None: avoid = set()
    else: avoid = set(avoid)
    coords = []
    used = set()
    while len(coords) < count:
        x = random.randint(0, max_x - 1)
        y = random.randint(0, max_y - 1)
        if (x, y) not in used and (x, y) not in avoid:
            used.add((x, y))
            coords.append((x, y))
    return coords

def draw_grid(screen):
    for x in range(0, WINDOW_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, Color.WHITE.value, (x, 0), (x, WINDOW_HEIGHT), 1)
    for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, Color.WHITE.value, (0, y), (WINDOW_WIDTH, y), 1)

def draw_block(screen, x, y, color, label=""):
    px, py = x * GRID_SIZE, y * GRID_SIZE
    pygame.draw.rect(screen, color.value, (px, py, GRID_SIZE, GRID_SIZE))
    if label:
        font = pygame.font.SysFont(None, int(GRID_SIZE * 0.5))
        text = font.render(label, True, Color.WHITE.value)
        rect = text.get_rect(center=(px + GRID_SIZE//2, py + GRID_SIZE//2))
        screen.blit(text, rect)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("地图初始化展示（无匹配路径）")
    

    stops = generate_random_xy(COUNT_STOP, GRID_COLS, GRID_ROWS)
    users = generate_random_xy(COUNT_USER, GRID_COLS, GRID_ROWS, avoid=stops)
    
    user_dests = {}
    temp_avoid = stops + users
    for u in users:
        d = generate_random_xy(1, GRID_COLS, GRID_ROWS, avoid=temp_avoid)[0]
        user_dests[u] = d
        temp_avoid.append(d)
        
    cars = generate_random_xy(COUNT_CAR, GRID_COLS, GRID_ROWS, avoid=temp_avoid)


    screen.fill(Color.BLACK.value)
    draw_grid(screen)
    
    for (x, y) in stops: draw_block(screen, x, y, Color.RED, "X")
    for i, (x, y) in enumerate(users): draw_block(screen, x, y, Color.GREEN, f"U{i}")
    for i, (ux, uy) in enumerate(user_dests): 
        dx, dy = user_dests[(ux, uy)]
        draw_block(screen, dx, dy, Color.PURPLE, f"D{i}")
    for i, (x, y) in enumerate(cars): draw_block(screen, x, y, Color.BLUE, f"C{i}")

    pygame.display.flip()


    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.time.delay(100)

if __name__ == "__main__":
    main()
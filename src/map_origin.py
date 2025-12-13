import folium
import random
import tkinter as tk
from tkinter import messagebox
import webbrowser
import os

# 南京市中心经纬度（基准点）
NANJING_CENTER = (32.041544, 118.777977)
# 南京市区经纬度范围（大致）
LAT_RANGE = (31.9, 32.2)    # 纬度范围
LNG_RANGE = (118.5, 119.0)  # 经度范围

# 障碍物坐标（模拟南京部分地标/建筑作为障碍物）
OBSTACLES = [
    (32.0288, 118.7732),  # 南京站
    (31.9569, 118.7850),  # 南京南站
    (32.0603, 118.8026),  # 玄武湖
    (32.0472, 118.7904),  # 新街口
    (31.9896, 118.7828),  # 夫子庙
    (32.1024, 118.8087),  # 紫金山
    (31.9361, 118.7723),  # 雨花台
    (32.0782, 118.8266),  # 红山森林动物园
]

def generate_random_coords():
    """生成南京范围内的随机经纬度"""
    lat = random.uniform(LAT_RANGE[0], LAT_RANGE[1])
    lng = random.uniform(LNG_RANGE[0], LNG_RANGE[1])
    return (lat, lng)

def create_taxi_map(user_count, driver_count):
    """创建打车地图"""
    # 初始化地图（以南京市中心为中心点）
    m = folium.Map(
        location=NANJING_CENTER,
        zoom_start=11,
        tiles='OpenStreetMap'  # 使用开源街道地图
    )

    # 添加障碍物标记（红色八角形）
    for idx, (lat, lng) in enumerate(OBSTACLES):
        folium.Marker(
            location=[lat, lng],
            popup=f"障碍物 {idx+1}<br>坐标：{lat:.6f}, {lng:.6f}",
            icon=folium.Icon(color='red', icon='ban', prefix='fa')
        ).add_to(m)

    # 生成并添加用户标记（蓝色人形图标）
    users = []
    for i in range(user_count):
        coords = generate_random_coords()
        users.append(coords)
        folium.Marker(
            location=coords,
            popup=f"打车用户 {i+1}<br>坐标：{coords[0]:.6f}, {coords[1]:.6f}",
            icon=folium.Icon(color='blue', icon='user', prefix='fa')
        ).add_to(m)

    # 生成并添加司机标记（绿色汽车图标）
    drivers = []
    for i in range(driver_count):
        coords = generate_random_coords()
        drivers.append(coords)
        folium.Marker(
            location=coords,
            popup=f"出租车司机 {i+1}<br>坐标：{coords[0]:.6f}, {coords[1]:.6f}",
            icon=folium.Icon(color='green', icon='car', prefix='fa')
        ).add_to(m)

    # 保存地图文件
    map_file = "nanjing_taxi_map.html"
    m.save(map_file)
    
    # 返回生成的信息
    return {
        "map_file": map_file,
        "user_count": user_count,
        "driver_count": driver_count,
        "obstacle_count": len(OBSTACLES),
        "users": users,
        "drivers": drivers,
        "obstacles": OBSTACLES
    }

def on_generate_click():
    """生成按钮点击事件"""
    try:
        # 获取输入的数量
        user_count = int(entry_user.get())
        driver_count = int(entry_driver.get())
        
        # 验证数量合法性
        if user_count < 1 or user_count > 1000:
            messagebox.showerror("错误", "用户数量必须在1-1000之间")
            return
        if driver_count < 1 or driver_count > 500:
            messagebox.showerror("错误", "司机数量必须在1-500之间")
            return
        
        # 生成地图
        result = create_taxi_map(user_count, driver_count)
        
        # 打开地图文件
        webbrowser.open('file://' + os.path.realpath(result["map_file"]))
        
        # 显示成功信息
        messagebox.showinfo(
            "成功",
            f"地图生成完成！\n"
            f"用户数量：{result['user_count']}\n"
            f"司机数量：{result['driver_count']}\n"
            f"障碍物数量：{result['obstacle_count']}\n"
            f"地图文件已保存为：{result['map_file']}"
        )
        
    except ValueError:
        messagebox.showerror("错误", "请输入有效的数字")
    except Exception as e:
        messagebox.showerror("错误", f"生成地图时出错：{str(e)}")

# 创建GUI界面
if __name__ == "__main__":
    # 主窗口设置
    root = tk.Tk()
    root.title("南京打车地图生成器")
    root.geometry("400x200")
    root.resizable(False, False)
    
    # 创建标签和输入框
    label_user = tk.Label(root, text="用户数量：")
    label_user.grid(row=0, column=0, padx=10, pady=20, sticky="e")
    
    entry_user = tk.Entry(root, width=20)
    entry_user.grid(row=0, column=1, padx=10, pady=20)
    entry_user.insert(0, "50")  # 默认值
    
    label_driver = tk.Label(root, text="司机数量：")
    label_driver.grid(row=1, column=0, padx=10, pady=5, sticky="e")
    
    entry_driver = tk.Entry(root, width=20)
    entry_driver.grid(row=1, column=1, padx=10, pady=5)
    entry_driver.insert(0, "20")  # 默认值
    
    # 生成按钮
    btn_generate = tk.Button(
        root, 
        text="生成打车地图", 
        command=on_generate_click,
        width=20,
        height=2,
        bg="#4CAF50",
        fg="white"
    )
    btn_generate.grid(row=2, column=0, columnspan=2, pady=20)
    
    # 运行主循环
    root.mainloop()
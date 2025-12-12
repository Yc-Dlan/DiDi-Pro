import random
import json
import math
from datetime import datetime, timedelta
from typing import List, Dict

# -------------------------- 打车订单核心配置（可自定义） --------------------------
# 生成订单总数
ORDER_NUM = 20
# 目标城市经纬度范围
CITY_LON_RANGE = (121.38, 121.55)  # 经度
CITY_LAT_RANGE = (31.18, 31.35)    # 纬度
# 订单时间范围（模拟一天7:00-23:00的打车场景）
BASE_DATE = datetime(2025, 1, 1)
# 乘客数上限
MAX_PASSENGER = 4
# 拼车订单比例
CARPOOL_RATIO = 0.2  # 20%拼车订单
# 立即用车/预约用车比例
IMMEDIATE_RATIO = 0.85  # 85%立即用车
# 上下车点距离限制（公里）- 仅用于校验合理性，不输出
MIN_DISTANCE_KM = 0.5   # 最小距离（避免过近订单）
MAX_DISTANCE_KM = 30    # 最大距离（避免超远订单）

# -------------------------- 工具函数 --------------------------
def cal_km_by_lon_lat(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """经纬度转换为公里数(WGS84坐标系) - 仅用于校验上下车点距离合理性"""
    lon1_rad, lat1_rad = math.radians(lon1), math.radians(lat1)
    lon2_rad, lat2_rad = math.radians(lon2), math.radians(lat2)
    d_lon = lon2_rad - lon1_rad
    d_lat = lat2_rad - lat1_rad
    a = math.sin(d_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return round(6371.0 * c, 2)

# -------------------------- 打车订单类（仅生成逻辑） --------------------------
class TaxiOrder:
    """打车订单生成类，仅包含订单核心属性和生成规则"""
    # 类级别的计数器，用于生成递增订单ID
    _order_counter = 1

    def __init__(self):
        # 基础标识（改为递增ID：order 1, order 2...）
        self.order_id: str = self._generate_order_id()
        self.create_time: datetime = self._generate_create_time()
        
        # 地理位置（上下车点）
        self.start_lon: float = round(random.uniform(*CITY_LON_RANGE), 6)
        self.start_lat: float = round(random.uniform(*CITY_LAT_RANGE), 6)
        self.end_lon: float = 0.0
        self.end_lat: float = 0.0
        
        # 乘客
        self.passenger_num: int = 0  # 乘客数
        self.is_carpool: bool = random.random() < CARPOOL_RATIO  # 是否拼车
        
        # 用车类型
        self.use_type: str = "立即用车" if random.random() < IMMEDIATE_RATIO else "预约用车"
        self.departure_time: datetime = self._init_departure_time()  # 出发时间

        self._generate_end_location()
        self._init_passenger_num()

    def _generate_order_id(self) -> str:
        """生成递增的唯一订单ID"""
        order_id = f"order {self._order_counter}"
        self.__class__._order_counter += 1
        return order_id

    def _generate_create_time(self) -> datetime:
        """随机生成下单时间（7:00-23:00）"""
        random_hour = random.randint(7, 22)
        random_minute = random.randint(0, 59)
        random_second = random.randint(0, 59)
        return BASE_DATE.replace(hour=random_hour, minute=random_minute, second=random_second)

    def _generate_end_location(self):
        """生成下车点，保证距离在合理范围（仅校验，不保存距离）"""
        while True:
            end_lon = round(random.uniform(*CITY_LON_RANGE), 6)
            end_lat = round(random.uniform(*CITY_LAT_RANGE), 6)
            distance = cal_km_by_lon_lat(self.start_lon, self.start_lat, end_lon, end_lat)
            if MIN_DISTANCE_KM <= distance <= MAX_DISTANCE_KM:
                self.end_lon = end_lon
                self.end_lat = end_lat
                break

    def _init_passenger_num(self):
        """初始化乘客数（统一使用上限）"""
        self.passenger_num = random.randint(1, MAX_PASSENGER)

    def _init_departure_time(self) -> datetime:
        """初始化出发时间（立即用车=下单时间，预约=未来时间）"""
        if self.use_type == "立即用车":
            return self.create_time
        else:
            # 预约时间：下单后10分钟~24小时
            random_delta = timedelta(minutes=random.randint(10, 1440))
            return self.create_time + random_delta

    def to_dict(self) -> Dict:
        """转换为字典，用于JSON序列化（删除指定字段）"""
        return {
            "order_id": self.order_id,
            "create_time": self.create_time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_location": {"lon": self.start_lon, "lat": self.start_lat},
            "end_location": {"lon": self.end_lon, "lat": self.end_lat},
            "passenger_num": self.passenger_num,
            "is_carpool": self.is_carpool,
            "use_type": self.use_type,
            "departure_time": self.departure_time.strftime("%Y-%m-%d %H:%M:%S")
        }

# -------------------------- 订单生成与导出 --------------------------
def generate_taxi_orders(num: int) :
    """生成指定数量的打车订单"""
    TaxiOrder._order_counter = 1
    orders = []
    for _ in range(num):
        orders.append(TaxiOrder())
    return orders

def save_taxi_orders(orders, file_path: str = "taxi_orders.json"):
    """将订单保存为JSON文件"""
    orders_dict = [order.to_dict() for order in orders]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(orders_dict, f, ensure_ascii=False, indent=4)
    print(f"✅ 成功生成 {len(orders)} 条打车订单，文件保存至：{file_path}")

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 生成订单
    taxi_orders = generate_taxi_orders(ORDER_NUM)
    
    # 保存为JSON
    save_taxi_orders(taxi_orders)


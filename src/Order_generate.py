import random
import json
import math
from typing import List, Dict
from Distance_transfer import cal_km_by_lon_lat

# 生成订单总数
ORDER_NUM = 100
# 目标城市经纬度范围(南京市，且与车辆随机生成的经纬度范围相同)
CITY_LON_RANGE = (118.22, 119.14)  # 经度
CITY_LAT_RANGE = (31.14, 32.37)    # 纬度
# 乘客数上限
MAX_PASSENGER = 4
# 拼车订单比例
CARPOOL_RATIO = 0.2  
# 拼车时最大乘客数
MAX_CARPOOL_PASSENGER = 2
# 上下车点距离限制
MIN_DISTANCE_KM = 0.5  
MAX_DISTANCE_KM = 30    

class TaxiOrder:
    """打车订单生成类，仅包含订单核心属性（无时间/用车类型相关逻辑）"""
    _order_counter = 1

    def __init__(self):
        self.order_id: str = self._generate_order_id()
        
        # 地理位置（上下车点）
        self.start_lon = round(random.uniform(*CITY_LON_RANGE), 6)
        self.start_lat = round(random.uniform(*CITY_LAT_RANGE), 6)
        self.end_lon = 0.0
        self.end_lat = 0.0
        
        # 乘客与拼车
        self.passenger_num = 0  # 乘客数
        self.is_carpool = random.random() < CARPOOL_RATIO  # 是否拼车

        # 初始化下车点和乘客数
        self._generate_end_location()
        self._init_passenger_num()

    def _generate_order_id(self):
        """生成递增的唯一订单ID"""
        order_id = f"order {self._order_counter}"
        self.__class__._order_counter += 1
        return order_id

    def _generate_end_location(self):
        """生成下车点，保证距离在合理范围"""
        while True:
            end_lon = round(random.uniform(*CITY_LON_RANGE), 6)
            end_lat = round(random.uniform(*CITY_LAT_RANGE), 6)
            distance = cal_km_by_lon_lat(self.start_lon, self.start_lat, end_lon, end_lat)
            if MIN_DISTANCE_KM <= distance <= MAX_DISTANCE_KM:
                self.end_lon = end_lon
                self.end_lat = end_lat
                break

    def _init_passenger_num(self):
        """初始化乘客数"""
        if self.is_carpool:
            # 拼车订单
            self.passenger_num = random.randint(1, MAX_CARPOOL_PASSENGER)
        else:
            # 非拼车订单
            self.passenger_num = random.randint(1, MAX_PASSENGER)

def generate_taxi_orders(num: int):
    """生成指定数量的打车订单"""
    TaxiOrder._order_counter = 1
    orders = []
    for _ in range(num):
        orders.append(TaxiOrder())
    return orders


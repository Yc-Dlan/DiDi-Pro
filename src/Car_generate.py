import random
import json
import math
from typing import List, Dict

# 生成车辆总数
CAR_NUM = 20  
# 目标城市经纬度范围（上海核心区）
CITY_LON_RANGE = (121.38, 121.55)  # 经度
CITY_LAT_RANGE = (31.18, 31.35)    # 纬度
# 经纬度精度（小数点后6位，符合GPS真实精度）
COORDINATE_PRECISION = 6
# 司机ID前缀（模拟真实司机编号）
DRIVER_ID_PREFIX = "DRV_"

class NetCarLocation:
    """网约车位置生成类，仅生成车辆ID、司机ID和经纬度位置数据"""
    # 类级别的计数器，用于生成递增车辆ID
    _car_counter = 1

    def __init__(self):
        # 基础标识
        self.car_id: str = self._generate_car_id()          # 车辆ID
        self.driver_id: str = self._generate_driver_id()    # 所属司机ID
        
        # 核心位置信息
        self.lon: float = round(random.uniform(*CITY_LON_RANGE), COORDINATE_PRECISION)  # 车辆经度
        self.lat: float = round(random.uniform(*CITY_LAT_RANGE), COORDINATE_PRECISION)  # 车辆纬度

    def _generate_car_id(self):
        """生成递增的唯一车辆ID（car 1, car 2...）"""
        car_id = f"car {self._car_counter}"
        self.__class__._car_counter += 1
        return car_id

    def _generate_driver_id(self):
        """生成随机司机ID（模拟真实编号）"""
        random_suffix = ''.join(random.choices('0123456789ABCDEF', k=8))
        return f"{DRIVER_ID_PREFIX}{random_suffix}"


def generate_netcar_locations(num):
    """生成指定数量的网约车位置数据（仅含ID和经纬度）"""
    NetCarLocation._car_counter = 1
    locations = []
    for _ in range(num):
        locations.append(NetCarLocation())
    return locations


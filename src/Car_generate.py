import random
import json
import math
from typing import List, Dict

# 生成车辆总数
CAR_NUM = 20
# 目标城市经纬度范围（南京市）
CITY_LON_RANGE = (118.22, 119.14)  # 经度
CITY_LAT_RANGE = (31.14, 32.37)    # 纬度
# 经纬度精度
COORDINATE_PRECISION = 6
# 司机ID前缀
DRIVER_ID_PREFIX = "DRV_"

class NetCarLocation:
    """网约车位置生成类,生成车辆ID、司机ID和经纬度位置数据"""
    _car_counter = 1

    def __init__(self):
        self.car_id = self._generate_car_id()          # 车辆ID
        self.driver_id = self._generate_driver_id()    # 所属司机ID
        
        self.lon = round(random.uniform(*CITY_LON_RANGE), COORDINATE_PRECISION)  # 车辆经度
        self.lat = round(random.uniform(*CITY_LAT_RANGE), COORDINATE_PRECISION)  # 车辆纬度

    def _generate_car_id(self):
        """生成车辆ID"""
        car_id = f"car {self._car_counter}"
        self.__class__._car_counter += 1
        return car_id

    def _generate_driver_id(self):
        """生成随机司机ID"""
        random_suffix = ''.join(random.choices('0123456789ABCDEF', k=8))
        return f"{DRIVER_ID_PREFIX}{random_suffix}"

def generate_netcar_locations(num):
    """生成指定数量的网约车位置数据"""
    NetCarLocation._car_counter = 1
    locations = []
    for _ in range(num):
        locations.append(NetCarLocation())
    return locations


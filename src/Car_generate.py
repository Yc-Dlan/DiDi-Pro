import random
import json
import math
from datetime import datetime, timedelta
from typing import List, Dict


# -------------------------- 网约车位置核心配置（可自定义） --------------------------
# 生成车辆总数
CAR_NUM = 200  
# 目标城市经纬度范围（和订单生成一致：上海核心区）
CITY_LON_RANGE = (121.38, 121.55)  # 经度
CITY_LAT_RANGE = (31.18, 31.35)    # 纬度
# 位置更新时间范围（模拟一天7:00-23:00的运营时段）
BASE_DATE = datetime(2025, 1, 1)
# 车辆状态比例（总和=1）
CAR_STATUS_RATIO = {
    "空载": 0.4,    # 空车可接单
    "接单中": 0.3,  # 已接单前往接乘客
    "已接单": 0.2,  # 正在送乘客
    "停运": 0.1     # 临时停运（休息/维保）
}
# 车辆在线比例
ONLINE_RATIO = 0.9  # 90%车辆在线运营
# 经纬度精度（小数点后6位，符合GPS真实精度）
COORDINATE_PRECISION = 6
# 司机ID前缀（模拟真实司机编号）
DRIVER_ID_PREFIX = "DRV_"

# -------------------------- 工具函数 --------------------------
def cal_km_by_lon_lat(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """经纬度转换为公里数(WGS84坐标系) - 用于校验位置合理性/后续距离计算"""
    lon1_rad, lat1_rad = math.radians(lon1), math.radians(lat1)
    lon2_rad, lat2_rad = math.radians(lon2), math.radians(lat2)
    d_lon = lon2_rad - lon1_rad
    d_lat = lat2_rad - lat1_rad
    a = math.sin(d_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return round(6371.0 * c, 2)

# -------------------------- 网约车位置类（仅生成逻辑） --------------------------
class NetCarLocation:
    """网约车位置生成类，模拟真实车辆实时位置数据"""
    # 类级别的计数器，用于生成递增车辆ID
    _car_counter = 1

    def __init__(self):
        # 基础标识
        self.car_id: str = self._generate_car_id()          # 车辆ID
        self.driver_id: str = self._generate_driver_id()    # 所属司机ID
        self.update_time: datetime = self._generate_update_time()  # 位置更新时间
        
        # 核心位置信息
        self.lon: float = round(random.uniform(*CITY_LON_RANGE), COORDINATE_PRECISION)  # 车辆经度
        self.lat: float = round(random.uniform(*CITY_LAT_RANGE), COORDINATE_PRECISION)  # 车辆纬度
        
        # 车辆状态
        self.is_online: bool = random.random() < ONLINE_RATIO  # 是否在线
        self.car_status: str = self._random_car_status()       # 车辆运营状态

    def _generate_car_id(self) -> str:
        """生成递增的唯一车辆ID（car 1, car 2...）"""
        car_id = f"car {self._car_counter}"
        self.__class__._car_counter += 1
        return car_id

    def _generate_driver_id(self) -> str:
        """生成随机司机ID（模拟真实编号）"""
        random_suffix = ''.join(random.choices('0123456789ABCDEF', k=8))
        return f"{DRIVER_ID_PREFIX}{random_suffix}"

    def _generate_update_time(self) -> datetime:
        """随机生成位置更新时间（7:00-23:00）"""
        random_hour = random.randint(7, 22)
        random_minute = random.randint(0, 59)
        random_second = random.randint(0, 59)
        return BASE_DATE.replace(hour=random_hour, minute=random_minute, second=random_second)

    def _random_car_status(self) -> str:
        """按比例随机生成车辆状态"""
        if not self.is_online:  # 离线车辆默认状态为停运
            return "停运"
        statuses = list(CAR_STATUS_RATIO.keys())
        ratios = list(CAR_STATUS_RATIO.values())
        return random.choices(statuses, weights=ratios, k=1)[0]

    def to_dict(self) -> Dict:
        """转换为字典，用于JSON序列化"""
        return {
            "car_id": self.car_id,
            "driver_id": self.driver_id,
            "update_time": self.update_time.strftime("%Y-%m-%d %H:%M:%S"),
            "location": {"lon": self.lon, "lat": self.lat},
            "is_online": self.is_online,
            "car_status": self.car_status
        }

# -------------------------- 位置生成与导出 --------------------------
def generate_netcar_locations(num: int) -> List[NetCarLocation]:
    """生成指定数量的网约车位置数据"""
    NetCarLocation._car_counter = 1
    locations = []
    for _ in range(num):
        locations.append(NetCarLocation())
    return locations

def save_netcar_locations(locations: List[NetCarLocation], file_path: str = "netcar_locations.json"):
    """将网约车位置数据保存为JSON文件"""
    locations_dict = [loc.to_dict() for loc in locations]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(locations_dict, f, ensure_ascii=False, indent=4)
    print(f"✅ 成功生成 {len(locations)} 条网约车位置数据，文件保存至：{file_path}")

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 生成网约车位置数据
    car_locations = generate_netcar_locations(CAR_NUM)
    
    # 保存为JSON
    save_netcar_locations(car_locations)

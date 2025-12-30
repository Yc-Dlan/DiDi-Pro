import math

def cal_km_by_lon_lat(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """经纬度转换为公里数(WGS84坐标系) - 用于校验位置合理性/后续距离计算"""
    lon1_rad, lat1_rad = math.radians(lon1), math.radians(lat1)
    lon2_rad, lat2_rad = math.radians(lon2), math.radians(lat2)
    d_lon = lon2_rad - lon1_rad
    d_lat = lat2_rad - lat1_rad
    a = math.sin(d_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return round(6371.0 * c, 2)

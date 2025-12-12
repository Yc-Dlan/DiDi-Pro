
import requests

def get_amap_distance(origins: str, destination: str, api_key: str = "a411f51400580ccf196c3799ddc1e1a4") -> float:
    """
    调用高德地图API计算两点间的直线距离（单位：米）
    :param origins: 起点经纬度，格式为 "经度,纬度"（如："116.481028,39.989643"）
    :param destination: 终点经纬度，格式为 "经度,纬度"（如："114.465302,40.004717"）
    :param api_key: 高德地图API密钥（默认使用你提供的key，建议替换为自己的）
    :return: 两点间的直线距离（单位：米，浮点数）
    :raises: ValueError - 参数格式错误/API返回失败；requests.exceptions.RequestException - 网络请求错误
    """
    # 1. 参数格式校验（检查是否包含","，且分割后是数字）
    def _check_lng_lat(lng_lat: str) -> bool:
        if "," not in lng_lat:
            return False
        lng, lat = lng_lat.split(",")
        try:
            float(lng)
            float(lat)
            return True
        except ValueError:
            return False

    if not _check_lng_lat(origins):
        raise ValueError(f"起点格式错误！请传入 '经度,纬度' 格式，当前值：{origins}")
    if not _check_lng_lat(destination):
        raise ValueError(f"终点格式错误！请传入 '经度,纬度' 格式，当前值：{destination}")

    # 2. 配置API请求参数
    url = "https://restapi.amap.com/v3/distance"  # 修正原URL的?parameters冗余
    params = {
        "key": api_key,
        "origins": origins,
        "destination": destination,
        "type": 1  # 0=直线距离，1=驾车距离（可根据需求调整）
    }

    try:
        # 3. 发送请求（设置超时时间，避免无限等待）
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # 触发HTTP状态码异常（如404/500）
        result = response.json()

        # 4. 解析返回结果
        if result.get("status") != "1":
            raise ValueError(f"API返回失败：{result.get('info', '未知错误')}")
        if not result.get("results"):
            raise ValueError("API未返回距离数据")

        # 5. 提取距离并转换为浮点数
        distance = float(result["results"][0]["distance"])
        return distance

    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"网络请求失败：{str(e)}")
    except Exception as e:
        raise Exception(f"未知错误：{str(e)}")


if __name__ == "__main__":
    # 示例1：正常调用
    try:
        start_point = "116.481028,39.989643"
        end_point = "114.465302,40.004717"
        distance = get_amap_distance(start_point, end_point)
        print(f"两点驾车距离：{distance:.2f} 米")  # 保留两位小数
    except Exception as e:
        print(f"计算距离失败：{e}")

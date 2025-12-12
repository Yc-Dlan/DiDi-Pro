#带有真实地图API
import requests

def address_to_lnglat(address: str, api_key: str = "a411f51400580ccf196c3799ddc1e1a4") -> tuple[float, float]:
    """
    调用高德地理编码API，将地址转换为高德坐标系的经纬度
    :param address: 要查询的地址（如："北京市朝阳区天安门"）
    :param api_key: 高德API密钥
    :return: (经度, 纬度)（浮点数元组）
    :raises: 异常（包含错误信息）
    """
    if not address.strip():
        raise ValueError("地址不能为空")

    # 高德地理编码API的URL
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {
        "key": api_key,
        "address": address,
        "output": "JSON"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        result = response.json()

        if result["status"] != "1":
            raise ValueError(f"地理编码失败：{result.get('info', '未知错误')}")
        if not result["geocodes"]:
            raise ValueError(f"未找到地址'{address}'对应的经纬度")

        # 提取经纬度（格式："经度,纬度"）
        lnglat_str = result["geocodes"][0]["location"]
        lng, lat = map(float, lnglat_str.split(","))
        return (lng, lat)

    except requests.exceptions.RequestException as e:
        raise Exception(f"网络请求失败：{str(e)}")
    except Exception as e:
        raise Exception(f"地址转经纬度失败：{str(e)}")




def lnglat_to_address(lng: float, lat: float, api_key: str = "a411f51400580ccf196c3799ddc1e1a4", radius: int = 1000) -> dict:
    """
    调用高德逆地理编码API，将经纬度转换为对应的地理名称（地址、行政区等）
    :param lng: 经度（浮点数，如116.481028）
    :param lat: 纬度（浮点数，如39.989643）
    :param api_key: 高德API密钥
    :param radius: 搜索半径（单位：米，默认1000米）
    :return: 地理信息字典（包含地址、省、市、区等）
    :raises: 异常（包含错误信息）
    """
    # 参数校验
    if not (-180 <= lng <= 180) or not (-90 <= lat <= 90):
        raise ValueError("经纬度超出合法范围（经度：-180~180，纬度：-90~90）")
    if radius < 1:
        raise ValueError("搜索半径不能小于1米")

    # 高德逆地理编码API地址
    url = "https://restapi.amap.com/v3/geocode/regeo"
    params = {
        "key": api_key,
        "location": f"{lng},{lat}",  # 格式："经度,纬度"
        "radius": radius,
        "output": "JSON",
        "extensions": "base"  # base=基础信息；all=详细信息（需额外权限）
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        result = response.json()

        if result["status"] != "1":
            raise ValueError(f"逆地理编码失败：{result.get('info', '未知错误')}")
        if result["regeocode"]["formatted_address"] == "":
            raise ValueError("未找到该经纬度对应的地理信息")

        # 提取关键信息（可根据需求扩展）
        regeo_info = result["regeocode"]
        address_data = {
            "完整地址": regeo_info["formatted_address"],
            "省": regeo_info["addressComponent"].get("province", ""),
            "市": regeo_info["addressComponent"].get("city", ""),
            "区/县": regeo_info["addressComponent"].get("district", ""),
            "街道": regeo_info["addressComponent"].get("streetNumber", {}).get("street", ""),
            "门牌号": regeo_info["addressComponent"].get("streetNumber", {}).get("number", "")
        }
        return address_data

    except requests.exceptions.RequestException as e:
        raise Exception(f"网络请求失败：{str(e)}")
    except Exception as e:
        raise Exception(f"经纬度转地理名称失败：{str(e)}")


# 调用示例
if __name__ == "__main__":
    try:
        # 示例：传入经纬度，获取地理名称
        lng, lat = 116.481028, 39.989643
        address_info = lnglat_to_address(lng, lat, api_key="a411f51400580ccf196c3799ddc1e1a4")
        print(str(lng)+","+str(lat)+"经纬度对应的地理信息：")
        for key, value in address_info.items():
            print(f"{key}：{value}")
    except Exception as e:
        print(e)




# # 调用示例
# if __name__ == "__main__":
#     try:
#         address = "北京市海淀区中关村"
#         lng, lat = address_to_lnglat(address, api_key="a411f51400580ccf196c3799ddc1e1a4")
#         print(f"地址'{address}'的经纬度：经度{lng:.6f}，纬度{lat:.6f}")
#     except Exception as e:
#         print(e)
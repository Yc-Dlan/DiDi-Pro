import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import sys
import os

from Car_generate import (
    NetCarLocation,
    generate_netcar_locations,
    CAR_NUM,
    CITY_LON_RANGE as CAR_LON_RANGE,
    CITY_LAT_RANGE as CAR_LAT_RANGE
)
from Order_generate import (
    TaxiOrder,
    generate_taxi_orders,
    ORDER_NUM,
    CITY_LON_RANGE as ORDER_LON_RANGE,
    CITY_LAT_RANGE as ORDER_LAT_RANGE
)

# -------------------------- Visualization Configuration (Customizable) --------------------------
VIS_CONFIG = {
    "figure_size": (14, 10),
    "car_status_style": {  # Vehicle status style (color/marker config)
        "Idle": {"color": "forestgreen", "marker": "o", "size": 60, "alpha": 0.8},
        "EnRouteToPickup": {"color": "gold", "marker": "s", "size": 60, "alpha": 0.8},
        "OnTrip": {"color": "crimson", "marker": "^", "size": 60, "alpha": 0.8},
        "NonOperational": {"color": "dimgray", "marker": "x", "size": 60, "alpha": 0.6}
    },
    "order_style": {  # Order start/end point style
        "start": {"color": "lightskyblue", "marker": ".", "size": 30, "alpha": 0.5},
        "end": {"color": "navy", "marker": ".", "size": 30, "alpha": 0.5},
        "line": {"color": "gray", "linewidth": 0.5, "alpha": 0.3}
    },
    "title": "Ride-hailing Car & Taxi Order Distribution (Normalized Coordinates)",
    "x_label": "Normalized Longitude (Origin = Min Longitude)",
    "y_label": "Normalized Latitude (Origin = Min Latitude)",
    "save_path": "car_order_normalized_plot.png",
    "font": {"family": "DejaVu Sans", "size": 10}  # English font config
}

# -------------------------- Core Utility Functions --------------------------
def collect_all_coordinates(
    cars: List[NetCarLocation],
    orders: List[TaxiOrder]
) -> Tuple[List[float], List[float]]:
    """
    Collect all longitude/latitude points (car locations + order start/end points) 
    for calculating global normalization boundaries
    :param cars: List of car data objects
    :param orders: List of order data objects
    :return: List of all longitudes, List of all latitudes
    """
    all_lons = []
    all_lats = []

    # Collect car coordinates
    for car in cars:
        all_lons.append(car.lon)
        all_lats.append(car.lat)

    # Collect order start/end coordinates
    for order in orders:
        all_lons.append(order.start_lon)
        all_lats.append(order.start_lat)
        all_lons.append(order.end_lon)
        all_lats.append(order.end_lat)

    return all_lons, all_lats

def normalize_coords(
    lons: List[float],
    lats: List[float],
    global_min_lon: float,
    global_max_lon: float,
    global_min_lat: float,
    global_max_lat: float
) -> Tuple[List[float], List[float]]:
    """
    Normalize coordinates: map longitude/latitude to [0,1] range (origin = global minimum)
    :param lons: List of original longitudes
    :param lats: List of original latitudes
    :param global_min_lon: Global minimum longitude
    :param global_max_lon: Global maximum longitude
    :param global_min_lat: Global minimum latitude
    :param global_max_lat: Global maximum latitude
    :return: List of normalized longitudes, List of normalized latitudes
    """
    # Avoid division by zero (extreme case: all points have the same coordinates)
    lon_range = global_max_lon - global_min_lon if global_max_lon != global_min_lon else 1e-8
    lat_range = global_max_lat - global_min_lat if global_max_lat != global_min_lat else 1e-8

    # Normalization calculation
    norm_lons = [(lon - global_min_lon) / lon_range for lon in lons]
    norm_lats = [(lat - global_min_lat) / lat_range for lat in lats]

    return norm_lons, norm_lats

# -------------------------- Status Mapping (Chinese -> English) --------------------------
def map_car_status_cn_to_en(status_cn: str) -> str:
    """
    Map Chinese car status to English for consistent visualization
    :param status_cn: Chinese status string from Car_generate
    :return: Corresponding English status string
    """
    status_mapping = {
        "空载": "Idle",
        "接单中": "EnRouteToPickup",
        "已接单": "OnTrip",
        "停运": "NonOperational"
    }
    return status_mapping.get(status_cn, "NonOperational")

# -------------------------- Main Visualization Function --------------------------
def visualize_cars_orders():
    """
    Main visualization process: 
    Generate data → Calculate global normalization boundaries → Normalize coordinates → Plot and display
    """
    # 1. Generate car and order data (call imported functions)
    print("🔄 Generating car location data...")
    cars = generate_netcar_locations(CAR_NUM)
    print("🔄 Generating taxi order data...")
    orders = generate_taxi_orders(ORDER_NUM)

    # 2. Collect all coordinates and calculate global normalization boundaries
    print("📏 Calculating global longitude/latitude boundaries...")
    all_lons, all_lats = collect_all_coordinates(cars, orders)
    global_min_lon, global_max_lon = min(all_lons), max(all_lons)
    global_min_lat, global_max_lat = min(all_lats), max(all_lats)

    # 3. Normalize all coordinates
    # 3.1 Normalize car coordinates
    car_raw_lons = [car.lon for car in cars]
    car_raw_lats = [car.lat for car in cars]
    car_norm_lons, car_norm_lats = normalize_coords(
        car_raw_lons, car_raw_lats,
        global_min_lon, global_max_lon,
        global_min_lat, global_max_lat
    )

    # 3.2 Normalize order start/end coordinates
    order_start_lons = [order.start_lon for order in orders]
    order_start_lats = [order.start_lat for order in orders]
    order_end_lons = [order.end_lon for order in orders]
    order_end_lats = [order.end_lat for order in orders]

    start_norm_lons, start_norm_lats = normalize_coords(
        order_start_lons, order_start_lats,
        global_min_lon, global_max_lon,
        global_min_lat, global_max_lat
    )
    end_norm_lons, end_norm_lats = normalize_coords(
        order_end_lons, order_end_lats,
        global_min_lon, global_max_lon,
        global_min_lat, global_max_lat
    )

    # 4. Plot configuration
    plt.rcParams["font.sans-serif"] = [VIS_CONFIG["font"]["family"]]  # English font
    plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display
    fig, ax = plt.subplots(figsize=VIS_CONFIG["figure_size"])

    # 5. Plot orders (draw first to avoid occlusion by cars)
    print("🎨 Plotting order start/end points...")
    # 5.1 Plot order start points
    ax.scatter(
        start_norm_lons, start_norm_lats,
        color=VIS_CONFIG["order_style"]["start"]["color"],
        marker=VIS_CONFIG["order_style"]["start"]["marker"],
        s=VIS_CONFIG["order_style"]["start"]["size"],
        alpha=VIS_CONFIG["order_style"]["start"]["alpha"],
        label="Order Start Points"
    )
    # 5.2 Plot order end points
    ax.scatter(
        end_norm_lons, end_norm_lats,
        color=VIS_CONFIG["order_style"]["end"]["color"],
        marker=VIS_CONFIG["order_style"]["end"]["marker"],
        s=VIS_CONFIG["order_style"]["end"]["size"],
        alpha=VIS_CONFIG["order_style"]["end"]["alpha"],
        label="Order End Points"
    )
    # 5.3 Plot order path lines (optional)
    for i in range(len(orders)):
        ax.plot(
            [start_norm_lons[i], end_norm_lons[i]],
            [start_norm_lats[i], end_norm_lats[i]],
            color=VIS_CONFIG["order_style"]["line"]["color"],
            linewidth=VIS_CONFIG["order_style"]["line"]["linewidth"],
            alpha=VIS_CONFIG["order_style"]["line"]["alpha"]
        )

    # 6. Plot cars (group by status for clear legend)
    print("🎨 Plotting car locations...")
    # Create status grouping (English)
    car_status_groups = {}
    for idx, car in enumerate(cars):
        status_en = map_car_status_cn_to_en(car.car_status)
        if status_en not in car_status_groups:
            car_status_groups[status_en] = []
        car_status_groups[status_en].append(idx)

    # Plot each status group
    for status_en, indices in car_status_groups.items():
        status_lons = [car_norm_lons[i] for i in indices]
        status_lats = [car_norm_lats[i] for i in indices]
        style = VIS_CONFIG["car_status_style"].get(status_en, VIS_CONFIG["car_status_style"]["NonOperational"])
        
        ax.scatter(
            status_lons, status_lats,
            color=style["color"],
            marker=style["marker"],
            s=style["size"],
            alpha=style["alpha"],
            label=f"Car - {status_en} ({len(indices)} vehicles)"
        )

    # 7. Plot styling
    ax.set_title(VIS_CONFIG["title"], fontsize=16, pad=20)
    ax.set_xlabel(VIS_CONFIG["x_label"], fontsize=12)
    ax.set_ylabel(VIS_CONFIG["y_label"], fontsize=12)
    ax.legend(loc="best", fontsize=VIS_CONFIG["font"]["size"])
    ax.grid(True, alpha=0.3, linestyle="--")

    # 8. Save and display plot
    plt.tight_layout()
    plt.savefig(VIS_CONFIG["save_path"], dpi=150, bbox_inches="tight")
    print(f"✅ Visualization plot saved to: {VIS_CONFIG['save_path']}")
    print(f"\n📌 Normalization Boundary Info:")
    print(f"   Global Longitude Range: {global_min_lon:.6f} ~ {global_max_lon:.6f}")
    print(f"   Global Latitude Range: {global_min_lat:.6f} ~ {global_max_lat:.6f}")
    print(f"   Normalized Coordinate Range: [0, 1] × [0, 1] (Origin = ({global_min_lon:.6f}, {global_min_lat:.6f}))")
    plt.show()

# -------------------------- Program Entry --------------------------
if __name__ == "__main__":
    visualize_cars_orders()

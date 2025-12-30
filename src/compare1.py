import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, DefaultDict
from collections import defaultdict
import random
from Order_generate import TaxiOrder, generate_taxi_orders, ORDER_NUM
from Car_generate import NetCarLocation, generate_netcar_locations, CAR_NUM
from Distance_transfer import cal_km_by_lon_lat
from K_means import TaxiCarClusterMatcher

# ===================== Your Original PSO Core Class (UNCHANGED, FULL ENGLISH) =====================
class PSOOrderMatcher:
    """Particle Swarm Optimization for Order Dispatching (Original Algorithm)"""
    def __init__(
        self,
        subgroup_orders: List[TaxiOrder],
        subgroup_cars: List[NetCarLocation],
        w: float = 0.7,
        c1: float = 1.2,
        c2: float = 1.2,
        max_iter: int = 100,
        pop_size: int = 50,
        k_distance: float = 0.001,
        k_imbalance: float = 0.5,
        k_carpool: float = 0.8,
        k_unassigned: float = 1.0,
        weight_distance: float = 0.5,
        weight_imbalance: float = 0.2,
        weight_carpool: float = 0.2,
        weight_unassigned: float = 0.1,
        empty_weight: float = 1.5
    ):
        self.orders = subgroup_orders
        self.cars = subgroup_cars
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.k_distance = k_distance
        self.k_imbalance = k_imbalance
        self.k_carpool = k_carpool
        self.k_unassigned = k_unassigned
        self.weight_distance = weight_distance
        self.weight_imbalance = weight_imbalance
        self.weight_carpool = weight_carpool
        self.weight_unassigned = weight_unassigned
        self.empty_weight = empty_weight
        self.car_states = {
            car.car_id: {"current_lon": car.lon, "current_lat": car.lat, "assigned_orders": [], "total_passengers": 0}
            for car in self.cars
        }
        self.particles = []
        self.pbest = []
        self.gbest = None
        self.pbest_fitness = []
        self.gbest_fitness = float('inf')
        self.gbest_fitness_history = []
        self.order_ids = [o.order_id for o in self.orders]
        self.car_ids = [c.car_id for c in self.cars]
        self.n_orders = len(self.orders)
        self.n_cars = len(self.cars)
        self.order_id_map = {o.order_id: o for o in self.orders}
        self.avg_orders_per_car = self.n_orders / self.n_cars if self.n_cars > 0 else 0
        self.max_distance_ref = self._calc_max_distance_ref()
        self.max_imbalance_ref = self.n_orders
        self.max_carpool_ref = 4
        self.max_unassigned_ref = self.n_orders

    def _calc_max_distance_ref(self) -> float:
        if self.n_orders == 0: return 1.0
        total = 0.0
        for order in self.orders:
            if not self.car_ids: continue
            empty_max = cal_km_by_lon_lat(self.car_states[self.car_ids[0]]["current_lon"],self.car_states[self.car_ids[0]]["current_lat"],order.start_lon, order.start_lat)*2
            ride = cal_km_by_lon_lat(order.start_lon, order.start_lat,order.end_lon, order.end_lat)
            total += empty_max * self.empty_weight + ride
        return total

    def _normalize(self, x: float, k: float, max_ref: float) -> float:
        if max_ref == 0 or x <= 0: return 0.0
        normalized_x = x / max_ref
        return 1 - np.exp(-k * normalized_x)

    def _initialize_particles(self):
        for _ in range(self.pop_size):
            particle = {o.order_id: np.random.choice(self.car_ids) for o in self.orders}
            self.particles.append(particle)
            self.pbest.append(particle.copy())
            self.pbest_fitness.append(float('inf'))

    def _calculate_fitness(self, particle: Dict[str, str]) -> float:
        total_weighted_distance = 0.0;total_imbalance=0.0;total_carpool_exceed=0.0;total_unassigned=0.0
        car_order_map = defaultdict(list)
        for oid, cid in particle.items(): car_order_map[cid].append(oid)
        temp_car_states = {k: {**v} for k, v in self.car_states.items()}
        assigned_orders = set()
        for car_id, order_ids in car_order_map.items():
            car_state = temp_car_states[car_id]
            curr_lon, curr_lat = car_state["current_lon"], car_state["current_lat"]
            curr_passengers = 0;car_assigned = []
            for oid in order_ids:
                order = self.order_id_map[oid]
                empty_dist = cal_km_by_lon_lat(curr_lon, curr_lat, order.start_lon, order.start_lat)
                ride_dist = cal_km_by_lon_lat(order.start_lon, order.start_lat, order.end_lon, order.end_lat)
                total_weighted_distance += empty_dist * self.empty_weight + ride_dist
                if order.is_carpool:
                    exceed = max(0, curr_passengers + order.passenger_num - 4)
                    total_carpool_exceed += exceed
                    if exceed == 0: curr_passengers += order.passenger_num;car_assigned.append(oid);assigned_orders.add(oid)
                else: curr_passengers = order.passenger_num;car_assigned.append(oid);assigned_orders.add(oid)
                curr_lon, curr_lat = order.end_lon, order.end_lat
            imbalance = max(0, len(car_assigned) - self.avg_orders_per_car)
            total_imbalance += imbalance
        total_unassigned = self.n_orders - len(assigned_orders)
        norm_d = self._normalize(total_weighted_distance, self.k_distance, self.max_distance_ref)
        norm_i = self._normalize(total_imbalance, self.k_imbalance, self.max_imbalance_ref)
        norm_c = self._normalize(total_carpool_exceed, self.k_carpool, self.max_carpool_ref)
        norm_u = self._normalize(total_unassigned, self.k_unassigned, self.max_unassigned_ref)
        return norm_d*self.weight_distance + norm_i*self.weight_imbalance + norm_c*self.weight_carpool + norm_u*self.weight_unassigned

    def _update_velocity_position(self, particle_idx: int):
        current_p = self.particles[particle_idx];pbest_p = self.pbest[particle_idx]
        for oid in self.order_ids:
            r1, r2 = np.random.random(), np.random.random()
            cog_prob = self.w * self.c1 * r1;soc_prob = self.w * self.c2 * r2
            if cog_prob > 0.5: current_p[oid] = pbest_p[oid]
            if soc_prob > 0.5 and self.gbest is not None: current_p[oid] = self.gbest[oid]

    def optimize(self) -> Tuple[Dict[str, str], float]:
        self._initialize_particles();self.gbest_fitness_history = []
        for _ in range(self.max_iter):
            current_gbest = float('inf');current_gbest_particle = None
            for i in range(self.pop_size):
                fitness = self._calculate_fitness(self.particles[i])
                if fitness < self.pbest_fitness[i]: self.pbest_fitness[i] = fitness;self.pbest[i] = self.particles[i].copy()
                if fitness < current_gbest: current_gbest = fitness;current_gbest_particle = self.particles[i].copy()
            if current_gbest < self.gbest_fitness: self.gbest_fitness = current_gbest;self.gbest = current_gbest_particle
            self.gbest_fitness_history.append(self.gbest_fitness)
            for i in range(self.pop_size): self._update_velocity_position(i)
        return self.gbest, self.gbest_fitness

# ===================== Core: Unified Distance Calculation Function (Fair Comparison for 3 Methods) =====================
def calculate_total_distance(matching_result: Dict[str, str], 
                             all_orders: List[TaxiOrder], 
                             all_cars: List[NetCarLocation]) -> Tuple[float, float, float, float]:
    """
    Unified core metric calculation for all 3 methods (100% same logic)
    Input: matching_result {order_id: car_id}, all orders, all cars
    Output: total_distance, empty_distance, ride_distance, empty_rate(%)
    Core Rule: Vehicle's next starting point = the end point of last completed order
    """
    car_order_map = defaultdict(list)
    for order_id, car_id in matching_result.items():
        car_order_map[car_id].append(order_id)
    
    total_all_dist = 0.0   # Total Travel Distance (Core Comparison Metric)
    total_empty_dist = 0.0 # Empty Distance (Vehicle -> Order Start Point)
    total_ride_dist = 0.0  # Ride Distance (Order Start -> Order End Point)
    
    for car_id, order_ids in car_order_map.items():
        car = next(c for c in all_cars if c.car_id == car_id)
        curr_lon, curr_lat = car.lon, car.lat # Vehicle initial position
        
        for order_id in order_ids:
            order = next(o for o in all_orders if o.order_id == order_id)
            # Calculate empty driving distance
            empty_dist = cal_km_by_lon_lat(curr_lon, curr_lat, order.start_lon, order.start_lat)
            # Calculate ride driving distance
            ride_dist = cal_km_by_lon_lat(order.start_lon, order.start_lat, order.end_lon, order.end_lat)
            
            # Accumulate distance
            total_empty_dist += empty_dist
            total_ride_dist += ride_dist
            total_all_dist += empty_dist + ride_dist
            
            # Update vehicle position: critical logic
            curr_lon, curr_lat = order.end_lon, order.end_lat
    
    # Calculate empty driving rate
    empty_rate = (total_empty_dist / total_all_dist) * 100 if total_all_dist > 0 else 0.0
    return round(total_all_dist,2), round(total_empty_dist,2), round(total_ride_dist,2), round(empty_rate,2)

# ===================== Implementation of 3 Assignment Methods =====================
def random_global_assign(orders: List[TaxiOrder], cars: List[NetCarLocation]) -> Dict[str, str]:
    """Method 1: Random Global Assignment - No clustering, random assign all orders to all cars"""
    car_ids = [c.car_id for c in cars]
    matching = {o.order_id: random.choice(car_ids) for o in orders}
    return matching

def random_cluster_assign(cluster_subgroups: Dict[int, Tuple[List[TaxiOrder], List[NetCarLocation]]]) -> Dict[str, str]:
    """Method 2: Random Cluster Assignment - K-means clustering first, then random assign in each subgroup"""
    total_matching = {}
    for sub_id, (sub_orders, sub_cars) in cluster_subgroups.items():
        sub_car_ids = [c.car_id for c in sub_cars]
        sub_matching = {o.order_id: random.choice(sub_car_ids) for o in sub_orders}
        total_matching.update(sub_matching)
    return total_matching

def pso_cluster_assign(cluster_subgroups: Dict[int, Tuple[List[TaxiOrder], List[NetCarLocation]]]) -> Dict[str, str]:
    """Method 3: Clustering + PSO Optimization - Your original optimal algorithm"""
    total_matching = {}
    for sub_id, (sub_orders, sub_cars) in cluster_subgroups.items():
        if len(sub_orders) == 0 or len(sub_cars) ==0: continue
        pso_matcher = PSOOrderMatcher(sub_orders, sub_cars)
        sub_matching, _ = pso_matcher.optimize()
        total_matching.update(sub_matching)
    return total_matching

# ===================== Visualization Comparison Function (100% English Chart) =====================
def plot_compare_result(compare_data: Dict):
    """Plot bar chart for 3 methods comparison: Total Distance + Empty Rate"""
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,6))
    methods = list(compare_data.keys())
    # Extract metrics
    total_dists = [compare_data[m]['Total Distance (km)'] for m in methods]
    empty_rates = [compare_data[m]['Empty Rate (%)'] for m in methods]
    colors = ['#ff7f7f','#7fbf7f','#7f7fff'] # Red, Green, Blue
    
    # Subplot 1: Total Travel Distance (Core Metric, Lower = Better)
    bars1 = ax1.bar(methods, total_dists, color=colors, alpha=0.8, width=0.6)
    ax1.set_title('3 Methods - Total Travel Distance Comparison (Core Indicator)', fontsize=14, pad=20)
    ax1.set_ylabel('Total Travel Distance (km)', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    # Add value labels
    for bar, val in zip(bars1, total_dists):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, f'{val}', ha='center', va='bottom', fontsize=11)
    
    # Subplot 2: Empty Driving Rate (Lower = Better)
    bars2 = ax2.bar(methods, empty_rates, color=colors, alpha=0.8, width=0.6)
    ax2.set_title('3 Methods - Empty Driving Rate Comparison', fontsize=14, pad=20)
    ax2.set_ylabel('Empty Driving Rate (%)', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    # Add value labels
    for bar, val in zip(bars2, empty_rates):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val}%', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.show()

# ===================== Main Function: Execute Comparative Experiment =====================
def main():
    # 1. Generate fixed orders & cars (same data for 3 methods, absolute fair)
    random.seed(666) # Fixed random seed for reproducible results
    np.random.seed(666)
    all_orders = generate_taxi_orders(ORDER_NUM)
    all_cars = generate_netcar_locations(CAR_NUM)
    print(f"✅ Experiment Initialized | Total Orders: {len(all_orders)} | Total Vehicles: {len(all_cars)}")
    print(f"{'='*80}")

    # 2. Execute K-means Clustering (for Method2 & Method3)
    cluster_matcher = TaxiCarClusterMatcher(n_clusters=4)
    cluster_matcher.cluster_taxi_orders(all_orders)
    cluster_matcher.cluster_netcar_locations(all_cars)
    cluster_matcher.match_cluster_centers()
    cluster_subgroups = cluster_matcher.split_into_subgroups()

    # 3. Run 3 assignment methods to get matching results
    print("✅ Start running three assignment methods...")
    match1 = random_global_assign(all_orders, all_cars)       # Method1: Random Global
    match2 = random_cluster_assign(cluster_subgroups)        # Method2: Random Cluster
    match3 = pso_cluster_assign(cluster_subgroups)           # Method3: Clustering + PSO

    # 4. Calculate core metrics with unified function
    res1 = calculate_total_distance(match1, all_orders, all_cars)
    res2 = calculate_total_distance(match2, all_orders, all_cars)
    res3 = calculate_total_distance(match3, all_orders, all_cars)

    # 5. Organize comparison data
    compare_result = {
        "Random Global": {
            "Total Distance (km)": res1[0],
            "Empty Distance (km)": res1[1],
            "Ride Distance (km)": res1[2],
            "Empty Rate (%)": res1[3]
        },
        "Random Cluster": {
            "Total Distance (km)": res2[0],
            "Empty Distance (km)": res2[1],
            "Ride Distance (km)": res2[2],
            "Empty Rate (%)": res2[3]
        },
        "Clustering + PSO": {
            "Total Distance (km)": res3[0],
            "Empty Distance (km)": res3[1],
            "Ride Distance (km)": res3[2],
            "Empty Rate (%)": res3[3]
        }
    }

    # 6. Print detailed comparison report
    print("📊 3 Assignment Methods - Core Performance Metrics Comparison (Same Dataset)")
    print(f"{'='*80}")
    print(f"{'Method':<18} | {'Total Dist(km)':<12} | {'Empty Dist(km)':<12} | {'Ride Dist(km)':<12} | {'Empty Rate(%)':<8}")
    print(f"{'='*80}")
    for method, data in compare_result.items():
        print(f"{method:<18} | {data['Total Distance (km)']:<12} | {data['Empty Distance (km)']:<12} | {data['Ride Distance (km)']:<12} | {data['Empty Rate (%)']:<8}")
    print(f"{'='*80}")

    # 7. Calculate optimization rate (prove PSO superiority)
    pso_vs_global = round((compare_result['Random Global']['Total Distance (km)'] - compare_result['Clustering + PSO']['Total Distance (km)'])/compare_result['Random Global']['Total Distance (km)']*100,2)
    pso_vs_cluster = round((compare_result['Random Cluster']['Total Distance (km)'] - compare_result['Clustering + PSO']['Total Distance (km)'])/compare_result['Random Cluster']['Total Distance (km)']*100,2)
    print(f"✅ Optimization Rate Comparison:")
    print(f"   PSO vs Random Global: Total Distance Reduced by {pso_vs_global}%")
    print(f"   PSO vs Random Cluster: Total Distance Reduced by {pso_vs_cluster}%")

    # 8. Generate visualization chart
    plot_compare_result(compare_result)

if __name__ == "__main__":
    main()

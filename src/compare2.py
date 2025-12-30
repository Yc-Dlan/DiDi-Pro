import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, DefaultDict
from collections import defaultdict
import random
from Order_generate import TaxiOrder, generate_taxi_orders, ORDER_NUM
from Car_generate import NetCarLocation, generate_netcar_locations, CAR_NUM
from Distance_transfer import cal_km_by_lon_lat
from K_means import TaxiCarClusterMatcher

# ===================== PSO Core Class (Unchanged Logic, Hyperparameter Tuning Adaptation) =====================
class PSOOrderMatcher:
    """Particle Swarm Optimization for Order Dispatching (Hyperparameter Tuning Version)"""
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

# ===================== Unified Performance Calculation Function (Fair Comparison) =====================
def calculate_performance(matching_result: Dict[str, str], all_orders: List[TaxiOrder], all_cars: List[NetCarLocation]) -> Dict:
    """Unified Calculation: Total Distance, Empty Rate, Order Completion Rate, Empty Distance, Ride Distance"""
    car_order_map = defaultdict(list)
    for oid, cid in matching_result.items(): car_order_map[cid].append(oid)
    
    total_all_dist = 0.0
    total_empty_dist = 0.0
    total_ride_dist = 0.0
    total_orders = len(all_orders)
    assigned_orders = len(matching_result)
    order_complete_rate = round((assigned_orders / total_orders)*100, 2) if total_orders>0 else 0.0

    for car_id, order_ids in car_order_map.items():
        car = next(c for c in all_cars if c.car_id == car_id)
        curr_lon, curr_lat = car.lon, car.lat
        for oid in order_ids:
            order = next(o for o in all_orders if o.order_id == oid)
            empty_dist = cal_km_by_lon_lat(curr_lon, curr_lat, order.start_lon, order.start_lat)
            ride_dist = cal_km_by_lon_lat(order.start_lon, order.start_lat, order.end_lon, order.end_lat)
            total_empty_dist += empty_dist
            total_ride_dist += ride_dist
            total_all_dist += empty_dist + ride_dist
            curr_lon, curr_lat = order.end_lon, order.end_lat

    empty_rate = round((total_empty_dist / total_all_dist)*100, 2) if total_all_dist>0 else 0.0
    return {
        "Total Distance (km)": round(total_all_dist, 2),
        "Empty Distance (km)": round(total_empty_dist, 2),
        "Ride Distance (km)": round(total_ride_dist, 2),
        "Empty Rate (%)": empty_rate,
        "Order Completion Rate (%)": order_complete_rate
    }

# ===================== Run PSO with Specific Hyperparameters =====================
def run_pso_with_params(cluster_subgroups, w, c1, c2):
    """Run PSO dispatch with specific hyperparameters, return matching result + fitness + performance"""
    total_matching = {}
    total_fitness = 0.0
    subgroup_count = 0
    for sub_id, (sub_orders, sub_cars) in cluster_subgroups.items():
        if len(sub_orders) ==0 or len(sub_cars)==0: continue
        pso = PSOOrderMatcher(sub_orders, sub_cars, w=w, c1=c1, c2=c2)
        sub_match, sub_fitness = pso.optimize()
        total_matching.update(sub_match)
        total_fitness += sub_fitness
        subgroup_count +=1
    avg_fitness = round(total_fitness / subgroup_count, 4) if subgroup_count>0 else 0.0
    return total_matching, avg_fitness

# ===================== Visualization for PSO Hyperparameter Tuning (100% English Chart) =====================
def plot_param_tune_result(tune_data):
    """Plot Hyperparameter Tuning Result: Total Travel Distance + Empty Rate Comparison (English Only)"""
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(18,7))
    
    param_labels = [f"w={d['w']}\nc1={d['c1']}\nc2={d['c2']}" for d in tune_data]
    total_dists = [d['Total Distance (km)'] for d in tune_data]
    empty_rates = [d['Empty Rate (%)'] for d in tune_data]
    colors = plt.cm.Set3(np.linspace(0, 1, len(param_labels)))

    # Subplot 1: Total Travel Distance (Core Indicator, Lower = Better)
    bars1 = ax1.bar(param_labels, total_dists, color=colors, alpha=0.8, width=0.6)
    ax1.set_title('PSO Hyperparameter Combinations - Total Travel Distance Comparison (Core Indicator)', fontsize=14, pad=20)
    ax1.set_ylabel('Total Travel Distance (km)', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, val in zip(bars1, total_dists):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3, f'{val}', ha='center', va='bottom', fontsize=9)

    # Subplot 2: Empty Driving Rate (Lower = Better)
    bars2 = ax2.bar(param_labels, empty_rates, color=colors, alpha=0.8, width=0.6)
    ax2.set_title('PSO Hyperparameter Combinations - Empty Driving Rate Comparison', fontsize=14, pad=20)
    ax2.set_ylabel('Empty Driving Rate (%)', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, val in zip(bars2, empty_rates):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

# ===================== Print PSO Hyperparameter Tuning Analysis Table (English) =====================
def print_param_tune_table(tune_data):
    """Print PSO Hyperparameter Tuning Analysis Table - Structured, Copy to Excel Directly"""
    print("\n" + "="*130)
    print("📊 PSO Hyperparameter Tuning - Performance Analysis Table (Same Dataset | Lower Indicator = Better Performance)")
    print("="*130)
    print(f"{'No.':<4} | {'w(Inertia)':<10} | {'c1(Cognitive)':<12} | {'c2(Social)':<10} | {'Total Dist(km)':<14} | {'Empty Rate(%)':<10} | {'Completion Rate(%)':<14} | {'Opt Fitness':<12}")
    print("="*130)
    for idx, data in enumerate(tune_data, 1):
        print(f"{idx:<4} | {data['w']:<10} | {data['c1']:<12} | {data['c2']:<10} | {data['Total Distance (km)']:<14} | {data['Empty Rate (%)']:<10} | {data['Order Completion Rate (%)']:<14} | {data['Opt Fitness']:<12}")
    print("="*130)

    # Filter the optimal hyperparameter combination (Min Total Distance = Best)
    best_param = min(tune_data, key=lambda x: x['Total Distance (km)'])
    print(f"✅ Recommended Optimal Hyperparameters: w={best_param['w']}, c1={best_param['c1']}, c2={best_param['c2']}")
    print(f"✅ Optimal Performance: Total Distance = {best_param['Total Distance (km)']}km, Empty Rate = {best_param['Empty Rate (%)']}%")
    print("="*130)

# ===================== Main: PSO Hyperparameter Tuning Experiment =====================
if __name__ == "__main__":
    # 1. Fixed random seed for reproducible & fair comparison (Critical!)
    random.seed(666)
    np.random.seed(666)
    all_orders = generate_taxi_orders(ORDER_NUM)
    all_cars = generate_netcar_locations(CAR_NUM)
    print(f"✅ PSO Hyperparameter Tuning Experiment Initialized")
    print(f"📌 Experiment Settings: Orders={len(all_orders)}, Vehicles={len(all_cars)}, Clusters=4, PSO Iter=100, Pop Size=50")

    # 2. K-means Clustering (Fixed for all hyperparameter combinations)
    cluster_matcher = TaxiCarClusterMatcher(n_clusters=4)
    cluster_matcher.cluster_taxi_orders(all_orders)
    cluster_matcher.cluster_netcar_locations(all_cars)
    cluster_matcher.match_cluster_centers()
    cluster_subgroups = cluster_matcher.split_into_subgroups()

    # 3. ✅ Key: PSO Hyperparameter Combinations for Tuning (Add/Delete Freely)
    pso_param_list = [
        (0.5, 1.0, 1.0),
        (0.5, 1.2, 1.2),
        (0.5, 1.5, 1.5),
        (0.7, 1.0, 1.0),
        (0.7, 1.2, 1.2),  # Your original default parameter
        (0.7, 1.5, 1.5),
        (0.8, 1.0, 1.0),
        (0.8, 1.2, 1.2),
        (0.8, 1.5, 1.5)
    ]

    # 4. Run all hyperparameter combinations and collect performance data
    param_tune_result = []
    print("\n🚀 Start PSO Hyperparameter Tuning Experiment, please wait...")
    for w, c1, c2 in pso_param_list:
        matching_res, fitness_val = run_pso_with_params(cluster_subgroups, w, c1, c2)
        perf_data = calculate_performance(matching_res, all_orders, all_cars)
        param_tune_result.append({
            "w": w,
            "c1": c1,
            "c2": c2,
            "Total Distance (km)": perf_data["Total Distance (km)"],
            "Empty Distance (km)": perf_data["Empty Distance (km)"],
            "Ride Distance (km)": perf_data["Ride Distance (km)"],
            "Empty Rate (%)": perf_data["Empty Rate (%)"],
            "Order Completion Rate (%)": perf_data["Order Completion Rate (%)"],
            "Opt Fitness": fitness_val
        })

    # 5. Generate Analysis Table + Visualization Chart
    print_param_tune_table(param_tune_result)
    plot_param_tune_result(param_tune_result)

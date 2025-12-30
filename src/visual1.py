import matplotlib.pyplot as plt
from Order_generate import generate_taxi_orders
from Car_generate import generate_netcar_locations

orders = generate_taxi_orders(100)
cars = generate_netcar_locations(20)

plt.scatter([o.start_lon for o in orders], [o.start_lat for o in orders], c='blue', label='Order Start', alpha=0.6)
plt.scatter([o.end_lon for o in orders], [o.end_lat for o in orders], c='green', label='Order End', alpha=0.6)
plt.scatter([c.lon for c in cars], [c.lat for c in cars], c='red', label='Cars', alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.title('Order and Car Distribution')
plt.show()

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

def get_routes(G, origin_point, destination_point, num_routes=5):
    # Get the nearest nodes to the origin and destination
    origin_node = ox.distance.nearest_nodes(G, origin_point[1], origin_point[0])
    destination_node = ox.distance.nearest_nodes(G, destination_point[1], destination_point[0])

    routes = []

    # 1. Shortest Route by Length
    routes.append(nx.shortest_path(G, origin_node, destination_node, weight='length'))

    # 2-5. Slightly different routes by tweaking edge weights to create diversity
    for i in range(4):
        for u, v, data in G.edges(data=True):
            data[f'length_variation_{i}'] = data['length'] * (1 + 0.05 * ((u + v + i) % (i + 3)))
        routes.append(nx.shortest_path(G, origin_node, destination_node, weight=f'length_variation_{i}'))

    return routes

def plot_routes(G, routes):
    ox.plot_graph_routes(G, routes, route_colors=['r', 'g', 'b', 'm', 'y'], route_linewidth=3, node_size=0)

if __name__ == "__main__":
    print("Loading Melbourne road network (this might take a few seconds)...")
    # Load Melbourne's drivable road network
    G = ox.graph_from_place('Melbourne, Australia', network_type='drive')

    print("\nEnter origin coordinates within Melbourne:")
    origin_lat = float(input("Origin Latitude: "))
    origin_lon = float(input("Origin Longitude: "))
    print("\nEnter destination coordinates within Melbourne:")
    dest_lat = float(input("Destination Latitude: "))
    dest_lon = float(input("Destination Longitude: "))

    origin = (origin_lat, origin_lon)
    destination = (dest_lat, dest_lon)

    routes = get_routes(G, origin, destination)

    print(f"\nGenerated {len(routes)} routes within Melbourne.")
    plot_routes(G, routes)
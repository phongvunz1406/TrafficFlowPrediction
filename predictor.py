import geopandas as gpd
import os
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from keras.models import load_model
from math import exp
import folium
from data.data import process_data
import random

# Load SCATS geojson
scats_gdf = gpd.read_file("data/Scats_Data.geojson")

# Load Melbourne road network graph
GRAPH_FILE = 'data/melbourne_graph.graphml'
if os.path.exists(GRAPH_FILE):
    G = ox.load_graphml(GRAPH_FILE)
    print("Graph loaded from cache.")


# Extract coordinates from SCATS
def get_coords_from_scats(user_input):
    # Auto-extract SCATS Number if user input is like '0970_HIGH_STREET_RD...'
    if "_" in user_input:
        user_input = user_input.split("_")[0]

    # Match against the correct SCATS_Number column
    match = scats_gdf[scats_gdf['SCATS_Number'].astype(str) == str(user_input)]
    if match.empty:
        match = scats_gdf[scats_gdf['Location'].str.contains(user_input, case=False, na=False)]
    if match.empty:
        return None
    row = match.iloc[0]
    return (row.geometry.y, row.geometry.x)

# Simplify graph to DiGraph with weights
G_simple = nx.DiGraph()
for u, v, data in G.edges(data=True):
    w = data.get('length', 1)
    if G_simple.has_edge(u, v):
        if G_simple[u][v]['weight'] > w:
            G_simple[u][v]['weight'] = w
    else:
        G_simple.add_edge(u, v, weight=w)

# Predict traffic flow using your model
def predict_traffic_flow(location, time, model_type):
    model_paths = {
        "GRU": r"model/gru.keras",
        "LSTM": r"model/lstm.keras",
        "SAES": r"model/saes.keras",
        "CNN": r"model/cnn.keras"
    }

    if model_type not in model_paths:
        model_type = "GRU"

    model_path = model_paths[model_type]
    try:
        model = load_model(model_path)
    except:
        print(f"Model '{model_type}' not found, using default flow.")
        return 100

    location = location.replace(" ", "_").replace("/", "_").replace("\\", "_")
    test_file = f'data/output_data/{location}.csv'

    if not os.path.exists(test_file):
        print(f"Missing traffic file: {test_file}")
        return 100

    _, _, X_test, _, scaler = process_data(test_file, test_file, 12)
    time_index = (pd.to_datetime(time) - pd.to_datetime('2006-10-26 00:00')).seconds // 900
    X_test_nn = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    predicted = model.predict(X_test_nn)
    predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
    return round(predicted[time_index])

# Extract street names from a route
def get_street_names(route):
    street_names = []
    print(f"\nExtracting street names for route with {len(route)} nodes")
    
    # Strategy 1: Direct edge data
    for i, (u, v) in enumerate(zip(route[:-1], route[1:])):
        try:
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                for key in edge_data:
                    if 'name' in edge_data[key]:
                        name = edge_data[key]['name']
                        if isinstance(name, list):
                            for street in name:
                                if street and street not in street_names:
                                    street_names.append(street)
                                    print(f"Found street: {street}")
                        elif name and name not in street_names:
                            street_names.append(name)
                            print(f"Found street: {name}")
        except:
            continue
    
    # Strategy 2: Check node neighbors if strategy 1 fails
    if not street_names:
        print("Strategy 1 failed. Trying neighbors lookup...")
        for node in route:
            try:
                for neighbor in G.neighbors(node):
                    for key, data in G[node][neighbor].items():
                        if 'name' in data:
                            name = data['name']
                            if isinstance(name, list):
                                for street in name:
                                    if street and street not in street_names:
                                        street_names.append(street)
                                        print(f"Found street: {street}")
                            elif name and name not in street_names:
                                street_names.append(name)
                                print(f"Found street: {name}")
            except:
                continue
    
    print(f"Found {len(street_names)} streets")
    if not street_names:
        street_names = ["Street names not available"]
    
    return street_names

# Create traffic-weighted graph
def create_traffic_weighted_graph(traffic_flow):
    print(f"Creating traffic-weighted graph with traffic flow: {traffic_flow}")
    G_traffic = G_simple.copy()
    
    # Calculate traffic factor
    base_speed = 60  # km/h
    F_base = max(traffic_flow * 0.5, 10)
    lambda_decay = 0.02
    excess_flow = max(traffic_flow - F_base, 0)
    speed = base_speed * exp(-lambda_decay * excess_flow)
    traffic_factor = 60 / speed if speed > 0 else 10
    
    print(f"Traffic factor: {traffic_factor} (speed: {speed} km/h)")
    
    # Update weights based on traffic
    for u, v in G_traffic.edges():
        G_traffic[u][v]['weight'] *= traffic_factor
    
    return G_traffic

# Generate 5 different routes
def generate_routes(G_traffic, origin_node, dest_node, max_routes=5):
    routes = []
    print(f"\nGenerating up to {max_routes} routes from {origin_node} to {dest_node}")
    
    # 1. Get shortest path first
    try:
        print("Finding shortest path...")
        shortest_route = nx.shortest_path(G_traffic, origin_node, dest_node, weight='weight')
        print(f"Found shortest path with {len(shortest_route)} nodes")
        routes.append(shortest_route)
    except nx.NetworkXNoPath:
        print("No path found between nodes")
        return []
    
    # 2. Generate alternatives by penalizing edges in existing routes
    print("Generating alternative routes by penalizing edges...")
    G_penalty = G_traffic.copy()
    
    for i in range(2):  # Try to get 2 routes using this method
        try:
            # Penalize edges in existing routes
            for route in routes:
                for u, v in zip(route[:-1], route[1:]):
                    if G_penalty.has_edge(u, v):
                        # Increase weight to make this edge less attractive
                        G_penalty[u][v]['weight'] *= 1.5
            
            # Find new path
            alt_route = nx.shortest_path(G_penalty, origin_node, dest_node, weight='weight')
            
            # Check if route is sufficiently different
            is_different = True
            for existing_route in routes:
                common_nodes = set(alt_route).intersection(set(existing_route))
                if len(common_nodes) > 0.7 * len(alt_route):
                    is_different = False
                    print("Route too similar to existing routes")
                    break
            
            if is_different:
                print(f"Found alternative route {len(routes)+1} with {len(alt_route)} nodes")
                routes.append(alt_route)
        except Exception as e:
            print(f"Error finding alternative route: {e}")
    
    # 3. Generate alternatives by adding random variation to weights
    if len(routes) < max_routes:
        print("Generating routes with randomized weights...")
        for i in range(max_routes - len(routes)):
            try:
                G_random = G_traffic.copy()
                
                # Add random variation to edge weights
                for u, v in G_random.edges():
                    G_random[u][v]['weight'] *= random.uniform(0.7, 1.3)
                
                # Find path with randomized weights
                rand_route = nx.shortest_path(G_random, origin_node, dest_node, weight='weight')
                
                # Check if route is sufficiently different
                is_different = True
                for existing_route in routes:
                    common_nodes = set(rand_route).intersection(set(existing_route))
                    if len(common_nodes) > 0.7 * len(rand_route):
                        is_different = False
                        break
                
                if is_different:
                    print(f"Found randomized route {len(routes)+1} with {len(rand_route)} nodes")
                    routes.append(rand_route)
            except Exception as e:
                print(f"Error finding randomized route: {e}")
    
    print(f"Generated {len(routes)} routes in total")
    return routes[:max_routes]

# Create map with multiple routes
def create_map(routes, origin_coord, dest_coord):
    print("Creating map with multiple routes")
    m = folium.Map(location=origin_coord, zoom_start=13)
    
    # Colors for different routes
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Add each route to the map
    for i, route in enumerate(routes):
        color = colors[i % len(colors)]
        
        # Get route coordinates
        try:
            route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
            
            # Add route to map
            folium.PolyLine(
                route_coords, 
                color=color, 
                weight=5, 
                opacity=0.7,
                tooltip=f"Route {i+1}"
            ).add_to(m)
            
            print(f"Added route {i+1} to map (color: {color})")
        except Exception as e:
            print(f"Error adding route {i+1} to map: {e}")
    
    # Add markers for origin and destination
    folium.Marker(location=origin_coord, tooltip="Origin", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(location=dest_coord, tooltip="Destination", icon=folium.Icon(color='red')).add_to(m)
    
    # Save the map
    m.save('osm_route_map.html')
    print("Map saved to osm_route_map.html")
    
    return m

# Main function to estimate travel times for multiple routes
def estimate_travel_time(origin_coord, dest_coord, date_time, model_choice, origin_input):
    print(f"\n===== ESTIMATING TRAVEL TIMES =====")
    print(f"From: {origin_coord}")
    print(f"To: {dest_coord}")
    print(f"Date/Time: {date_time}")
    print(f"Model: {model_choice}")
    
    # Find nearest nodes to coordinates
    origin_node = ox.distance.nearest_nodes(G, X=origin_coord[1], Y=origin_coord[0])
    dest_node = ox.distance.nearest_nodes(G, X=dest_coord[1], Y=dest_coord[0])
    print(f"Origin node: {origin_node}, Destination node: {dest_node}")
    
    # Predict traffic flow
    traffic_flow = predict_traffic_flow(origin_input, date_time, model_choice)
    print(f"Predicted traffic flow: {traffic_flow}")
    
    # Create traffic-weighted graph
    G_traffic = create_traffic_weighted_graph(traffic_flow)
    
    # Generate multiple routes
    routes = generate_routes(G_traffic, origin_node, dest_node)
    
    if not routes:
        return "No route found between origin and destination."
    
    # Calculate details for each route
    print("\nCalculating route details...")
    route_details = []
    
    for i, route in enumerate(routes):
        print(f"\nProcessing Route {i+1}...")
        
        # Calculate distance
        try:
            distance = sum(G_simple[u][v]['weight'] for u, v in zip(route[:-1], route[1:])) / 1000  # km
            
            # Calculate speed and time
            base_speed = 60  # km/h
            F_base = max(traffic_flow * 0.5, 10)
            lambda_decay = 0.02
            excess_flow = max(traffic_flow - F_base, 0)
            speed = base_speed * exp(-lambda_decay * excess_flow)
            time_mins = (distance / speed) * 60
            
            # Get street names for this route
            street_names = get_street_names(route)
            
            # Add route details
            route_details.append({
                'route_number': i + 1,
                'route': route,
                'distance': distance,
                'traffic_flow': traffic_flow,
                'speed': speed,
                'time': time_mins,
                'streets': street_names
            })
            
            print(f"Route {i+1} processed successfully")
        except Exception as e:
            print(f"Error processing route: {e}")
    
    # Create the map
    create_map([rd['route'] for rd in route_details], origin_coord, dest_coord)
    
    # Create result text
    result = f"From: {origin_coord}\n"
    result += f"To: {dest_coord}\n"
    result += f"Traffic Flow: {traffic_flow}\n\n"
    
    for rd in route_details:
        result += f"Route {rd['route_number']}:\n"
        result += f"  Distance: {rd['distance']:.2f} km\n"
        result += f"  Speed: {rd['speed']:.2f} km/h\n"
        result += f"  Estimated Time: {rd['time']:.0f} minutes\n"
        result += f"  Streets:\n"
        
        for street in rd['streets']:
            result += f"    - {street}\n"
        
        result += "\n"
    
    result += "Route map saved to osm_route_map.html\n"
    print("Result text generated successfully")
    
    return result
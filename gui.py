import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from tkcalendar import Calendar, DateEntry
from predictor import estimate_travel_time, get_coords_from_scats
from datetime import datetime
import webbrowser
import os
import threading
import http.server
import socketserver
import time

# Global variable to keep track of the server
map_server = None

def predict():
    # Disable the predict button to prevent multiple clicks
    predict_button.config(state=tk.DISABLED)
    
    origin_input = source_entry.get().strip()
    destination_input = dest_entry.get().strip()
    model = model_var.get()  

    date_str = date_entry.get()
    time_str = time_entry.get()

    # Validate fields
    if not origin_input or not destination_input or not date_str or not time_str:
        messagebox.showwarning("Input Error", "Please fill all the fields.")
        predict_button.config(state=tk.NORMAL)
        return

    # Combine date and time into a proper string
    date_time = f"{date_str} {time_str}:00"

    # Clear previous output
    output_text.delete('1.0', tk.END)
    
    # Get coordinates from SCATS
    origin_coord = get_coords_from_scats(origin_input)
    dest_coord = get_coords_from_scats(destination_input)

    if not origin_coord or not dest_coord:
        output_text.insert(tk.END, "Invalid origin or destination.\n")
        predict_button.config(state=tk.NORMAL)
        return

    # Update status
    status_var.set("Calculating routes...")
    
    # Show loading message
    output_text.insert(tk.END, "Calculating routes, please wait...\n")
    
    # Create a processing thread to keep UI responsive
    def calculation_thread():
        try:
            # Call the estimate_travel_time function
            result = estimate_travel_time(
                origin_coord, dest_coord, date_time, model, origin_input)
            
            # Schedule UI updates on the main thread
            root.after(0, lambda: update_ui(result))
            
        except Exception as error:
            # Schedule error display on the main thread
            root.after(0, lambda: display_error(str(error)))
    
    def update_ui(result):
        # Clear loading message
        output_text.delete('1.0', tk.END)
        
        # Display results in text area
        output_text.insert(tk.END, result)
        
        # Enable view map button
        view_map_button.config(state=tk.NORMAL)
        
        # Update status
        status_var.set("Ready")
        
        # Re-enable the predict button
        predict_button.config(state=tk.NORMAL)
    
    def display_error(error_message):
        output_text.delete('1.0', tk.END)
        output_text.insert(tk.END, f"Error: {error_message}\n")
        status_var.set("Error occurred")
        predict_button.config(state=tk.NORMAL)
    
    # Start the calculation in a separate thread
    threading.Thread(target=calculation_thread).start()

def view_map():
    """Open the map using a local HTTP server to ensure it displays correctly"""
    global map_server
    
    try:
        # Get the directory of the HTML file
        map_file = 'osm_route_map.html'
        map_dir = os.path.dirname(os.path.abspath(map_file))
        os.chdir(map_dir)
        
        # If there's already a server running, shut it down
        if map_server:
            try:
                map_server.shutdown()
                map_server.server_close()
            except:
                pass
        
        # Find an available port
        port = 8000
        Handler = http.server.SimpleHTTPRequestHandler
        
        # Create and start the server in a separate thread
        map_server = socketserver.TCPServer(("", port), Handler)
        server_thread = threading.Thread(target=map_server.serve_forever)
        server_thread.daemon = True  # Allow the thread to be terminated when the main program exits
        server_thread.start()
        
        # Give the server a moment to start
        time.sleep(0.5)
        
        # Update status
        status_var.set(f"Map server running on port {port}")
        
        # Open the map in the default web browser
        webbrowser.open(f'http://localhost:{port}/{map_file}')
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to display map: {str(e)}")
        status_var.set("Error displaying map")

def on_closing():
    """Handle application closing - shut down the map server if it's running"""
    global map_server
    if map_server:
        try:
            map_server.shutdown()
            map_server.server_close()
        except:
            pass
    root.destroy()

# Tkinter App Setup
root = tk.Tk()
root.title("Traffic Flow Prediction")
root.geometry("800x600")
root.protocol("WM_DELETE_WINDOW", on_closing)  # Set the closing handler

# Status bar
status_var = tk.StringVar()
status_var.set("Ready")
status_bar = tk.Label(root, textvariable=status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Create main content frame
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Input Frame
input_frame = tk.LabelFrame(main_frame, text="Input Parameters")
input_frame.pack(fill=tk.X, pady=5)

# Model Dropdown
tk.Label(input_frame, text="Select Model:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
model_var = tk.StringVar()
model_combo = ttk.Combobox(input_frame, textvariable=model_var)
model_combo['values'] = ("GRU", "LSTM", "SAES", "CNN")
model_combo.current(0)
model_combo.grid(row=0, column=1, sticky='w', padx=5, pady=5)

# Source input
tk.Label(input_frame, text="Source (e.g., 0970):").grid(row=1, column=0, sticky='w', padx=5, pady=5)
source_entry = tk.Entry(input_frame)
source_entry.grid(row=1, column=1, sticky='w', padx=5, pady=5)

# Destination input
tk.Label(input_frame, text="Destination (e.g., 3001):").grid(row=2, column=0, sticky='w', padx=5, pady=5)
dest_entry = tk.Entry(input_frame)
dest_entry.grid(row=2, column=1, sticky='w', padx=5, pady=5)

# Date selection
tk.Label(input_frame, text="Select Date:").grid(row=3, column=0, sticky='w', padx=5, pady=5)
date_entry = DateEntry(input_frame, width=12, background='darkblue',
                    foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
date_entry.grid(row=3, column=1, sticky='w', padx=5, pady=5)

# Time input
tk.Label(input_frame, text="Enter Time (HH:MM):").grid(row=4, column=0, sticky='w', padx=5, pady=5)
time_entry = tk.Entry(input_frame)
time_entry.insert(0, "10:30")
time_entry.grid(row=4, column=1, sticky='w', padx=5, pady=5)

# Predict Button
button_frame = tk.Frame(input_frame)
button_frame.grid(row=5, column=0, columnspan=2, pady=10)

predict_button = tk.Button(button_frame, text="Predict", command=predict)
predict_button.pack(side=tk.LEFT, padx=5)

view_map_button = tk.Button(button_frame, text="View Map", command=view_map, state=tk.DISABLED)
view_map_button.pack(side=tk.LEFT, padx=5)

# Output Area
output_frame = tk.LabelFrame(main_frame, text="Route Details")
output_frame.pack(fill=tk.BOTH, expand=True, pady=5)

# Add scrolled text for output
output_text = scrolledtext.ScrolledText(output_frame, height=20, wrap=tk.WORD)
output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

if __name__ == "__main__":
    root.mainloop()
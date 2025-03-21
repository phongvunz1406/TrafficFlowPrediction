import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import Calendar, DateEntry
from predictor import estimate_travel_time
from datetime import datetime

def predict():
    origin = source_entry.get().strip()
    destination = dest_entry.get().strip()
    model = model_var.get()
    date_str = date_entry.get()
    time_str = time_entry.get()

    if not origin or not destination or not date_str or not time_str:
        messagebox.showwarning("Input Error", "Please fill all the fields.")
        return

    # Combine date and time
    date_time = f"{date_str} {time_str}:00"

    # Clear previous output
    output_text.delete('1.0', tk.END)

    try:
        # Call your estimate_travel_time() function that RETURNS a result string
        result = estimate_travel_time(origin, destination, date_time, model)
        output_text.insert(tk.END, result)
    except Exception as e:
        output_text.insert(tk.END, f"Error: {str(e)}\n")

# Tkinter App Setup
root = tk.Tk()
root.title("Traffic Flow Prediction")
root.geometry("600x600")

# Model Dropdown
tk.Label(root, text="Select Model:").pack()
model_var = tk.StringVar()
model_combo = ttk.Combobox(root, textvariable=model_var)
model_combo['values'] = ("GRU", "LSTM", "SAES", "CNN")
model_combo.current(0)
model_combo.pack(pady=5)

# Source input
tk.Label(root, text="Source (e.g., 0970):").pack()
source_entry = tk.Entry(root)
source_entry.pack(pady=5)

# Destination input
tk.Label(root, text="Destination (e.g., 3001):").pack()
dest_entry = tk.Entry(root)
dest_entry.pack(pady=5)

# Date selection
tk.Label(root, text="Select Date:").pack()
date_entry = DateEntry(root, width=12, background='darkblue',
                    foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
date_entry.pack(pady=5)

# Time input
tk.Label(root, text="Enter Time (HH:MM):").pack()
time_entry = tk.Entry(root)
time_entry.insert(0, "10:30")
time_entry.pack(pady=5)

# Predict Button
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack(pady=10)

# Output Area
output_text = tk.Text(root, height=15, width=70)
output_text.pack(pady=10)

root.mainloop()

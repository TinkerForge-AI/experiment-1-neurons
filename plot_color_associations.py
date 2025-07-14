import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = "color_associations_over_time.csv"
csv_path = "training_logs/color_associations_over_time.csv"
if not os.path.exists(csv_path):
    print(f"File {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)
run_numbers = df['run']

# Find all color/neuron columns
color_neuron_cols = [col for col in df.columns if col != 'run']
colors = sorted(set(col.split('_')[0] for col in color_neuron_cols))
neuron_indices = sorted(set(int(col.split('_n')[1]) for col in color_neuron_cols))

plt.figure(figsize=(12, 6))
for neuron_idx in neuron_indices:
    for color in colors:
        col_name = f"{color}_n{neuron_idx}"
        if col_name in df:
            plt.plot(run_numbers, df[col_name], label=f"Neuron {neuron_idx} - {color}")

plt.xlabel("Run Number")
plt.ylabel("Color Success Count")
plt.title("Neuron Color Associations Over Time")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
# Save image to training_logs folder
plt.savefig("training_logs/color_association_evolution.png")
plt.close()

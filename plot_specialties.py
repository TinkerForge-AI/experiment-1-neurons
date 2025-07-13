import matplotlib.pyplot as plt
import pandas as pd

# Load specialty evolution CSV
csv_path = "specialties_over_time.csv"
df = pd.read_csv(csv_path)

plt.figure(figsize=(10, 6))
for i in range(10):
    plt.plot(df['run'], df[f'specialty_{i}'], label=f'Neuron {i}')
plt.xlabel('Run')
plt.ylabel('Specialty Value')
plt.title('Neuron Specialty Evolution Over Time')
plt.legend()
plt.tight_layout()
plt.show()

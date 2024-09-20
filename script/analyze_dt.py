import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

if len(sys.argv) != 2:
    print("Must pass csv path!")
    exit()

df = pd.read_csv(sys.argv[1])
# df = df[(df['last_ts'] >= 0.016) & (df['last_ts'] <= 0.28)]  # Remove outliers
df = df.loc[10:, :]  # Ignore very start.

mean = np.mean(df['last_ts'])
std = np.std(df['last_ts'])

print(f"mean: {mean}")
print(f"std: {std}")
print(f"max: {np.max(df['last_ts'])}")
print(f"min: {np.min(df['last_ts'])}")

# Create a boxplot of the 'last_ts' column
plt.figure(figsize=(8, 6))  # Set the figure size for better readability
plt.boxplot(df['last_ts'])
plt.title('Boxplot of dt')
plt.ylabel('dt values')
plt.grid(True)
plt.show()

# Plot histogram of the 'last_ts' column with normal
normal_ys = np.random.normal(loc=mean, scale=std, size=(df.shape[0], 1))
plt.hist(normal_ys, bins=1000, color='red', alpha=0.3)

plt.hist(df['last_ts'], bins=1000, color='blue', alpha=0.7)  # Adjust the number of bins for more granularity
plt.title('Histogram of dt')
plt.xlabel('dt values')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.scatter(df['steps'], df['last_ts'])
plt.xlabel("step")
plt.ylabel("last_ts")
plt.title("last_ts (s) over steps")
plt.show()

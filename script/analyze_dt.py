import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('data/with-dt.csv').iloc[1:, :]
mean = np.mean(df['dt'])
std = np.std(df['dt'])
df = df[(df['dt'] >= (mean - 3 * std)) & (df['dt'] <= (mean + 3 * std))]

print(f"mean: {mean}")
print(f"std: {std}")
print(f"max: {np.max(df['dt'])}")
print(f"min: {np.min(df['dt'])}")

# Create a boxplot of the 'dt' column
plt.figure(figsize=(8, 6))  # Set the figure size for better readability
plt.boxplot(df['dt'])
plt.title('Boxplot of dt')
plt.ylabel('dt values')
plt.grid(True)
plt.show()

# Plot histogram of the 'dt' column with normal
normal_ys = np.random.normal(loc=mean, scale=std, size=(df.shape[0], 1))
plt.hist(normal_ys, bins=1000, color='red', alpha=0.3)

plt.hist(df['dt'], bins=1000, color='blue', alpha=0.7)  # Adjust the number of bins for more granularity
plt.title('Histogram of dt')
plt.xlabel('dt values')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

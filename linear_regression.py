import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/life_expectancy_2013_2022_clean.csv")
x = df['income_index']
y = df['life_expectancy']

x_mean = np.mean(x)
y_mean = np.mean(y)

numerator =  ((x -x_mean) * (y - y_mean)).sum()
denominator = ((x - x_mean) ** 2).sum()

slope = numerator / denominator
intercept = y_mean - slope * x_mean

print(f"slope: {slope}, Intecept: {intercept}")
y_pred = slope * x + intercept

plt.scatter(x, y, label="Actual Data")
plt.plot(x, y_pred, color="red", label="Regression Line")
plt.xlabel("Income Index")
plt.ylabel("Life Expectancy")
plt.title("Linear Regression: Income Index vs. Life Expectancy")
plt.legend()
plt.savefig("Result/price_prediction_plot.png")
plt.show()

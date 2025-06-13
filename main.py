import pandas as pd
import matplotlib.pyplot as plt
from linear_model import LinearRegressionGD
from sklearn.metrics import mean_squared_error, r2_score


def main():
    df = pd.read_csv("data/life_expectancy_2013_2022_clean.csv")
    X = df[['income_index']]
    y = df['life_expectancy']

    model = LinearRegressionGD(lr=0.01, n_iter=1000)
    model.fit(X, y)
    y_pred = model.predict(X)

    print("📊 Model Performance:")
    print(f"Final MSE: {mean_squared_error(y, y_pred):.4f}")
    print(f"R² Score: {r2_score(y, y_pred):.4f}")

    plt.figure()
    plt.scatter(X, y, label="Actual Data")
    plt.plot(X, y_pred, color="red", label="Regression Line")
    plt.xlabel("Income Index")
    plt.ylabel("Life Expectancy")
    plt.title("Linear Regression Prediction")
    plt.legend()
    plt.savefig("results/prediction_plot.png")
    plt.show()


    plt.figure()
    plt.plot(model.mse_history)
    plt.xlabel("Iterations")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Learning Curve")
    plt.savefig("results/learning_curve.png")
    plt.show()

if __name__ == "__main__":
    main()
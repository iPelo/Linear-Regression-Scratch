# Linear Regression from Scratch – Life Expectancy Prediction

This project explores life expectancy using a linear regression model built from scratch. The goal is to understand how certain health and development indicators influence life expectancy across multiple countries from 2013 to 2022.

## 📊 Dataset Overview

The dataset is clean and simulated, inspired by real-world sources including:

- [World Bank Open Data](https://data.worldbank.org)
- [WHO Global Health Observatory](https://www.who.int/data/gho)
- [UNDP Human Development Reports](https://hdr.undp.org/data-center)

It focuses on six countries and includes yearly observations over a 10-year span.

### 🗂 Features (2013–2022)

| Feature                     | Description                                                  |
|----------------------------|--------------------------------------------------------------|
| `life_expectancy`          | Average life expectancy at birth                            |
| `infant_deaths`            | Infant deaths per 1,000 live births                         |
| `alcohol_per_capita_litres`| Alcohol consumption per capita (liters)                     |
| `health_exp_pct_gdp`       | Health expenditure as a % of GDP                            |
| `hepB_coverage_pct`        | Hepatitis B immunization coverage (%)                       |
| `literacy_pct`             | Adult literacy rate (%)                                     |
| `adult_mortality`          | Combined adult mortality (male and female)                  |
| `measles_cases_per_1000`   | Measles cases per 1,000 people                              |
| `income_index`             | Normalized income index (UNDP scale from 0 to 1)            |
| `developed_or_developing`  | Classification based on World Bank income level             |

## 🧮 Modeling Goal

The main objective is to predict life expectancy using features like income index, health expenditure, and alcohol use. This version uses gradient descent to optimize model weights and plots the learning curve for analysis.

The dataset was designed to reflect realistic distributions for learning and experimentation.

## 📌 Key Takeaways & Next Steps

### ✅ What This Project Demonstrates
- Implementation of linear regression using pure Python and NumPy
- Gradient descent optimization with visualized learning curve
- Ability to work with real-world inspired data
- Data visualization using matplotlib

### 🔜 Possible Extensions
- Add multiple features for multivariate regression
- Introduce a training/testing data split for better evaluation
- Compare results with `scikit-learn`'s LinearRegression model
- Explore regularization (L1/L2) to handle overfitting
- Include metrics like R² or MAE for more comprehensive evaluation

These improvements would increase model performance and demonstrate deeper machine learning proficiency.
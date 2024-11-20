import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Data Loading and Preprocessing
data = pd.read_csv('/content/Customer Purchasing Behaviors.csv')

# Handle missing values
data = data.dropna()

# Encode categorical variables
categorical_columns = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Data Splitting
X = data.drop(columns=['user_id', 'purchase_amount'])
y = data['purchase_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training and Evaluation
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'R^2': r2}

    print(f"{name} - RMSE: {rmse:.2f}, R^2: {r2:.2f}")

    # Predictions vs Actuals Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"Predictions vs Actual for {name}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()

# Model Comparison
results_df = pd.DataFrame(results).T
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.barplot(x=results_df.index, y='RMSE', data=results_df, ax=ax[0], palette="Blues")
ax[0].set_title("RMSE Comparison")
ax[0].set_ylabel("RMSE")

sns.barplot(x=results_df.index, y='R^2', data=results_df, ax=ax[1], palette="Greens")
ax[1].set_title("R^2 Score Comparison")
ax[1].set_ylabel("R^2 Score")

plt.tight_layout()
plt.show()

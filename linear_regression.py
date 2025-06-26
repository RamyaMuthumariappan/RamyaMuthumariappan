import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Housing.csv")  # use your local path if needed

# -------------------------
# SIMPLE LINEAR REGRESSION
# -------------------------
print("==== SIMPLE LINEAR REGRESSION ====\n")
X_simple = df[['area']]
y_simple = df['price']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)
model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)
y_pred_s = model_simple.predict(X_test_s)

# Evaluation
print("Intercept:", model_simple.intercept_)
print("Coefficient (area):", model_simple.coef_[0])
print("MAE:", mean_absolute_error(y_test_s, y_pred_s))
print("MSE:", mean_squared_error(y_test_s, y_pred_s))
print("R2 Score:", r2_score(y_test_s, y_pred_s))

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(X_test_s, y_test_s, color='blue', label='Actual')
plt.plot(X_test_s, y_pred_s, color='red', label='Regression Line')
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Simple Linear Regression: Area vs Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# MULTIPLE LINEAR REGRESSION
# -------------------------
print("\n==== MULTIPLE LINEAR REGRESSION ====\n")

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)
X_multi = df_encoded.drop('price', axis=1)
y_multi = df_encoded['price']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)

# Evaluation
print("Intercept:", model_multi.intercept_)
print("MAE:", mean_absolute_error(y_test_m, y_pred_m))
print("MSE:", mean_squared_error(y_test_m, y_pred_m))
print("R2 Score:", r2_score(y_test_m, y_pred_m))

# Coefficients
coefficients_multi = pd.Series(model_multi.coef_, index=X_multi.columns)
print("\nTop 5 Influential Features:")
print(coefficients_multi.sort_values(ascending=False).head(5))

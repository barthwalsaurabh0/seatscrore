import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
import pickle  # For saving the model

# Load the data
df = pd.read_csv("score.csv")

# Features and target
X = df[['Age', 'Gender', 'Fatigue']]
y = df['SeatScore']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create and train the regression tree
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open("seatscore_tree.pkl", "wb") as f:
    pickle.dump(model, f)

# Make predictions
y_pred = model.predict(X_test)

# Cap predictions between 0 and 100
y_pred_capped = np.clip(y_pred, 0, 100)

# Print predictions
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted (Raw)': y_pred,
    'Predicted (Capped)': y_pred_capped
})
print(results)

# Mean squared error
mse = mean_squared_error(y_test, y_pred_capped)
print(f"\nMean Squared Error: {mse:.2f}")

# Visualize the regression tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=['Age', 'Gender', 'Fatigue'], filled=True, rounded=True)
plt.title("Regression Tree for SeatScore")
plt.show()

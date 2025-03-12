import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset 
dataFrame = pd.read_excel("merc.xlsx")

# Display first 5 rows 
print(dataFrame.head())

# Check for missing values
print(dataFrame.isnull().sum())

# Drop rows with NaN values
dataFrame.dropna(inplace=True)

# Data visualization
plt.figure(figsize=(9, 3))
sns.histplot(dataFrame["price"], bins=30, kde=True)
plt.title("Price Distribution")
plt.show()

sns.countplot(x="year", data=dataFrame)
plt.xticks(rotation=45)
plt.title("Car Count by Year")
plt.show()

sns.scatterplot(x="mileage", y="price", data=dataFrame)
plt.title("Mileage vs Price")
plt.show()

# Correlation matrix
corr_matrix = dataFrame.corr(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Remove extreme outliers
filtered_df = dataFrame.sort_values("price", ascending=False).iloc[131:]

# Remove cars from 1970
filtered_df = filtered_df[filtered_df.year != 1970]

# Drop categorical column (if necessary)
if "transmission" in filtered_df.columns:
    filtered_df.drop("transmission", axis=1, inplace=True)

# Split dataset into X (features) and y (target)
y = filtered_df["price"].values
X = filtered_df.drop("price", axis=1).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Use transform instead of fit_transform

# Build neural network model
model = Sequential([
    Dense(12, activation="relu"),
    Dense(12, activation="relu"),
    Dense(12, activation="relu"),
    Dense(12, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# Train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=250, epochs=300)

# Convert training history to DataFrame and plot loss
loss_df = pd.DataFrame(history.history)
loss_df.plot()
plt.title("Model Training Loss Over Epochs")
plt.show()

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Scatter plot of actual vs predicted values
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Car Prices")
plt.show()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load dataset (replace 'data.csv' with your dataset path)
data = pd.read_csv('data.csv')

# Assume the dataset has columns: V (voltage), I (current), eff (efficiency), and target power (P)
# Replace column names with the actual names in your dataset
X = data[['V', 'I', 'eff']].values  # Features
y = data['P'].values  # Target variable (Power)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better neural network performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')  # Output layer for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    verbose=1
)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Absolute Error on Test Data: {mae}")

# Save the model for future use
model.save('iv_characteristics_model.h5')

# Example prediction
example_input = np.array([[12.5, 5.1, 0.85]])  # Replace with actual values
example_input_scaled = scaler.transform(example_input)
prediction = model.predict(example_input_scaled)
print(f"Predicted Power: {prediction[0][0]}")

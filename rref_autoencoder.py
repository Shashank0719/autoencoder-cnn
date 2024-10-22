import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor

# Function to compute the Reduced Row Echelon Form (RREF)
def rref_with_minus1(A):
    rows, cols = A.shape
    r = 0  # Row pointer
    for c in range(cols):
        if r >= rows:
            break
        # Find the pivot
        pivot = np.argmax(np.abs(A[r:, c])) + r
        if A[pivot, c] == 0:
            continue  # No pivot in this column
        # Swap rows
        A[[r, pivot]] = A[[pivot, r]]
        # Normalize the pivot row
        A[r] = A[r] / A[r, c]
        # Eliminate other rows
        for i in range(rows):
            if i != r:
                A[i] -= A[i, c] * A[r]
        r += 1
    
    # Apply -1 trick: replace zero rows with -1
    for i in range(r, rows):
        A[i] = np.where(A[i] == 0, -1, A[i])
    
    return A

data = pd.DataFrame({
    'Transaction_Amount': [100, 200, 300, 150, 4000],
    'Transaction_Type': ['A', 'B', 'A', 'C', 'B'],
    'Frequency': [5, 3, 1, 6, 2]
})

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

# Step 3: Define a preprocessing pipeline
numeric_features = ['Transaction_Amount', 'Frequency']
categorical_features = ['Transaction_Type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2, include_bias=True)),
    ('autoencoder', MLPRegressor(hidden_layer_sizes=(5,), activation='relu', max_iter=1000))
])

pipeline.fit(data)

transformed_features = pipeline.named_steps['preprocessor'].transform(data)
rref_matrix = rref_with_minus1(transformed_features.copy())
print("\nRREF with -1 Trick:\n", rref_matrix)

# Step 7: Get reconstruction error
reconstructed = pipeline.named_steps['autoencoder'].predict(transformed_features)
mse = np.mean((transformed_features - reconstructed) ** 2, axis=1)

print("\nReconstruction MSE for each transaction:\n", mse)

mean_error = np.mean(mse)
std_dev_error = np.std(mse)
threshold = mean_error + 2 * std_dev_error  # Example: mean + 2 standard deviations
print("\nDynamic Anomaly detection threshold:", threshold)

anomalies = mse > threshold
print("\nDetected Anomalies (True indicates anomaly):\n", anomalies)

anomalous_transactions = data[anomalies]
print("\nAnomalous Transactions:\n", anomalous_transactions)

### another approach

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['feature1', 'feature2']])

# Convert to DataFrame for easier interpretation
poly_feature_names = poly.get_feature_names_out(['feature1', 'feature2'])
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

# Combine with original target
poly_df['target'] = df['target']
print("\nPolynomial Features:")
print(poly_df)

# Step 3: Create Interaction Terms (if not included in PolynomialFeatures)
poly_df['feature1_feature2'] = poly_df['feature1'] * poly_df['feature2']

print("\nDataset with Interaction Terms:")
print(poly_df)

# Step 4: Apply RREF
X = poly_df.drop(columns='target').values
X_rref = rref_with_minus1(X.copy())  # Create a copy to avoid modifying original data

print("\nRREF Transformed Features:")
print(X_rref)

# Step 5: Anomaly Detection with Isolation Forest
model = IsolationForest(contamination=0.2, random_state=42)
model.fit(X_rref)

# Predict anomalies
predictions = model.predict(X_rref)

# Convert predictions: -1 (anomaly), 1 (normal)
predictions = np.where(predictions == -1, 1, 0)

# Add predictions to the DataFrame
poly_df['Anomaly_Prediction'] = predictions
print("\nAnomaly Detection Results:")
print(poly_df)

# Step 6: Visualization (optional)
plt.figure(figsize=(10, 6))
plt.scatter(poly_df['feature1'], poly_df['feature2'], c=poly_df['Anomaly_Prediction'], cmap='coolwarm', edgecolor='k')
plt.title('Anomaly Detection Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Anomaly Prediction (1: Anomaly, 0: Normal)')
plt.show()

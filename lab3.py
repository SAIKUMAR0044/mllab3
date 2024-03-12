import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Load data from the 'Purchase data' worksheet
file_path = 'C:\\Users\\sai jaswanth\\Desktop\\Machine Learning\\Lab Session1 Data.xlsx'
purchase_data = pd.read_excel(file_path, sheet_name='Purchase data')

# Extract matrices A and C
matrix_A = purchase_data.iloc[:, 1:4]
matrix_C = purchase_data.iloc[:, 4]

# Calculate vector space properties
dimensionality_vector_space = matrix_A.shape[1]
num_vectors_in_space = matrix_A.shape[0]

# Calculate the rank of Matrix A
rank_of_matrix_A = np.linalg.matrix_rank(matrix_A)

# Calculate the pseudo-inverse of Matrix A
pseudo_inverse_A = np.linalg.pinv(matrix_A)

# Calculate the cost of each product using the pseudo-inverse
product_costs = np.dot(pseudo_inverse_A, matrix_C)

# Display results
print("Matrix A:")
print(matrix_A)
print("Matrix C:")
print(matrix_C)
print("Dimensionality of the vector space:", dimensionality_vector_space)
print("Number of vectors in the vector space:", num_vectors_in_space)
print("Rank of Matrix A:", rank_of_matrix_A)
print("Cost of each product using Pseudo-Inverse:")
print(product_costs)

# Feature Engineering
purchase_data['Category'] = np.where(purchase_data['Payment (Rs)'] > 200, 'RICH', 'POOR')
features = purchase_data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
target = purchase_data['Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Choose a classifier (Random Forest as an example)
classifier = RandomForestClassifier()

# Train the model
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Day 18: Interactive Salary Predictor
# Concept: Simple Linear Regression with User Input
# Created by: Devendra Kumar Prajapat (Dev)

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Training Dataset
data = {
    'years_experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'salary': [30000, 35000, 42000, 48000, 55000, 62000, 70000, 81000, 88000, 95000]
}
df = pd.DataFrame(data)

# 2. Reshaping Features (X) and Target (y)
X = df[['years_experience']] # Independent Variable
y = df['salary']            # Dependent Variable

# 3. Model Training
model = LinearRegression()
model.fit(X, y)

print("--- ML Salary Prediction Model Trained Successfully ---")

# 4. Interactive User Input Section
try:
    print("\nCheck your predicted salary based on experience.")
    user_input = input("Enter Years of Experience: ").strip()
    
    # Convert input to float
    experience = float(user_input)
    
    # Prepare input for prediction (must be 2D array)
    input_data = np.array([[experience]])
    
    # Generate Prediction
    prediction = model.predict(input_data)
    
    # Output result formatted to 2 decimal places
    print("-" * 40)
    print(f"Experience Provided: {experience} years")
    print(f"Predicted Salary: ${prediction[0]:,.2f}")
    print("-" * 40)

except ValueError:
    print("Error: Invalid input. Please enter a numerical value for experience.")

print("Thank you for using the AI Predictor!")
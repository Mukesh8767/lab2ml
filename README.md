


Lab 02: Linear Regression Machine Learning Model
This repository contains the code and explanations for implementing a simple linear regression model with one variable. The lab demonstrates how to predict housing prices based on a single feature (such as square footage) using linear regression.

Objectives
Understand and implement a linear regression model with one variable.
Predict housing prices based on a single feature (e.g., square footage).
Create and evaluate the linear regression function 

f(w,b).
Make predictions using the trained model and assess its performance.
Files
Lab02_LinearRegression.ipynb: The Jupyter Notebook containing the linear regression implementation and explanation.
Prerequisites
To run the code in this repository, you need the following installed:

Python 3.x
Jupyter Notebook or Jupyter Lab
The following Python libraries:
numpy
pandas
matplotlib
scikit-learn
You can install these packages using pip:

bash
Copy code
pip install numpy pandas matplotlib scikit-learn
Usage
Clone the repository to your local machine:
bash
Copy code
git clone https://github.com/yourusername/linear-regression-lab02.git
Open the Jupyter notebook:
bash
Copy code
jupyter notebook Lab02_LinearRegression.ipynb
Run the cells in the notebook to execute the linear regression implementation.
Linear Regression Model Overview
The notebook covers the following key steps:

Data Loading: Load the housing dataset and extract the feature and target variable (e.g., square footage and price).
Model Function: Define the linear regression model 

f(w,b)=w×x+b, where 
𝑤
w is the weight, 
𝑏
b is the bias, and 
𝑥
x is the input feature.
Cost Function: Calculate the cost function (Mean Squared Error) to evaluate how well the model fits the data.
Gradient Descent: Optimize the model parameters (weight and bias) using gradient descent.
Model Training: Train the model using the data and update the parameters iteratively.
Prediction: Use the trained model to make predictions on new data (housing prices).
Performance Evaluation: Assess the performance of the model by evaluating metrics such as Mean Squared Error (MSE) and visualizing the predictions against actual data.
Example Output
After running the notebook, you will see the trained linear regression model parameters (w and b) and a plot showing the linear fit to the data. The notebook also includes predictions for new input values (e.g., square footage) and performance metrics.

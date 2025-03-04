# Linear Regression with Brute Force and Gradient Descent

## Overview  
This project demonstrates linear regression using two different approaches:  
1. **Brute-force scanning** to find the best slope (`m1`).  
2. **Gradient descent** for optimizing the slope efficiently.  

The dataset is generated using the equation:  
```
y = m1*x + m2 + {noise}
```

## Features  
- Generates a dataset with random noise.  
- Trains a linear regression model using Scikit-learn.  
- Computes and visualizes the **Mean Squared Error (MSE)**.  
- Finds the best slope (`m1`) using **brute-force scanning**.  
- Finds the optimal slope using **gradient descent**.  
- Compares the efficiency of both methods.  

## Dependencies  
Ensure you have the following Python libraries installed:  

```
pip install numpy pandas matplotlib scikit-learn
```
Usage
Run the script using:
```
python assign1.py
```
Functions and Their Purpose

### Functions and Their Purpose  

| Function | Description |
|----------|------------|
| `generate_data(n, m1, m2)` | Generates a dataset with `n` points. |
| `plot_data(x, y)` | Plots the dataset points. |
| `train_model(x, y)` | Trains a linear regression model. |
| `plot_best_fit(x, y, reg)` | Plots the best-fit regression line. |
| `compute_mse(y, y_pred)` | Computes Mean Squared Error (MSE). |
| `plot_mse(x, mse_list)` | Visualizes MSE for each point. |
| `plot_mse_vs_params(x, y, best_m1, best_m2)` | Plots MSE vs slope/intercept. |
| `brute_force_scan(m2, x, y, m1_range)` | Finds the best slope using brute force. |
| `gradient_descent(m2, x, y)` | Optimizes slope using gradient descent. |
| `efficiency(m1_range, losses_gd)` | Computes efficiency comparison. |



Gradient Descent is significantly more efficient, converging much faster than brute-force scanning.


Output
The script generates plots for:
✅ Data distribution
✅ Best-fit regression line
✅ MSE vs slope and intercept
✅ Loss curve for brute-force and gradient descent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import time

def generate_data(n=100, m1=3.1, m2=0.1):
    x = np.linspace(-10, 10, n)
    # print(x)
    noise = np.random.normal(0, 1, n) #generating n random noise values with mean=0 and standard deviation=1
    y = m1 * x + m2 + noise
    return x, y

def plot_data(x, y): #plotting the data points
    plt.scatter(x, y)
    plt.title("Dataset for (y = m1*x + m2 + noise)")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()

def train_model(x, y): 
    X = x.reshape(-1, 1) #-1 represents the no.of rows, it is a default value
    #1 represents column matrix
    reg = linear_model.LinearRegression()
    reg.fit(X, y)
    return reg

def plot_best_fit(x, y, reg): #plotting the best fit line
    X = x.reshape(-1, 1)
    plt.scatter(x, y)
    plt.plot(x, reg.predict(X), color='red')
    plt.title("Best Fit Line")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()


def compute_mse(y, y_pred):
    # mse=np.mean((y-y_pred)**2) #using mean function of statistics module
    # print("Mean Squared Error:",mse)
    mse_list = [(y_pred[i] - y[i]) ** 2 for i in range(len(y))]
    mse = sum(mse_list) / len(y) #finding mse without any inbuilt libraries
    return mse, mse_list

def plot_mse(x, mse_list): #function to plot the mse values
    plt.scatter(x, mse_list, color="blue")
    plt.plot(x, mse_list, color="black")
    plt.title("Mean Squared Error")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show() #Here it represents the mse error for each difference of y and y_pred

def plot_mse_vs_params(x, y, best_m1, best_m2, n=100):
    #plotting mse vs slope
    m1_range = np.linspace(0, 5, n)
    mse_m1 = [np.mean((y - (m1 * x + best_m2)) ** 2) for m1 in m1_range]
    plt.plot(m1_range, mse_m1, color="purple")
    plt.title("Mean Squared Error vs Slope")
    plt.xlabel("Slope")
    plt.ylabel("Mean Squared Error")
    plt.grid()
    plt.show()

    #plotting mse vs intercept
    m2_range = np.linspace(best_m2 - 2, best_m2 + 2, n)
    mse_m2 = [np.mean((y - (best_m1 * x + m2)) ** 2) for m2 in m2_range]
    plt.plot(m2_range, mse_m2, color="purple")
    plt.title("Mean Squared Error vs Intercept")
    plt.xlabel("Intercept")
    plt.ylabel("Mean Squared Error")
    plt.grid()
    plt.show()

def brute_force_scan(m2,x, y, m1_range):
    losses = []
    scanned = []
    start_time = time.time()
    for m in m1_range:
        y_pred = m * x + m2
        loss = np.mean((y - y_pred) ** 2)
        losses.append(loss)
        scanned.append((m, loss))
    end_time = time.time()
    best_m1 = m1_range[np.argmin(losses)] #np.argmin(losses) gives the index of the minimum of losses

    #plotting graph for linear scan
    plt.plot(m1_range, losses, label="Loss Curve", color="purple")
    plt.scatter([p[0] for p in scanned], [p[1] for p in scanned], label="Scanned Points", color="red")
    plt.axvline(best_m1, color='y', linestyle='dashed', label=f'Best m1: {best_m1:.2f}')
    plt.xlabel('m1 values')
    plt.ylabel('MSE Loss')
    plt.title('Brute-force Loss Scan for m1')
    plt.legend()
    plt.show()
    print("Brute-force Time taken:", end_time - start_time)
    return best_m1

def gradient_descent(m2,x, y, learning_rate=0.01, tolerance=1e-6): #determining minima using gradient descent
    n = len(x)
    m_gd = np.random.uniform(0, 5)
    losses_gd = []
    start_time = time.time()
    while True:
        y_pred = m_gd * x + m2
        md = -(2 / n) * sum(x * (y - y_pred))
        m_gd -= learning_rate * md
        loss = np.mean((y - y_pred) ** 2)
        losses_gd.append(loss)
        if len(losses_gd) > 1 and abs(losses_gd[-2] - losses_gd[-1]) < tolerance: #stopping criteria
            break
    end_time = time.time()
    #Plot gradient descent loss curve

    plt.plot(losses_gd, label="Gradient Descent Loss")
    plt.xlabel("Iterations")
    plt.ylabel("MSE Loss")
    plt.title("Loss Reduction with Gradient Descent")
    plt.legend()
    plt.show()
    print("Gradient Descent Time taken:", end_time - start_time)
    return m_gd, losses_gd

def efficiency(m1_range, losses_gd): #finding efficiency
    return len(m1_range) / len(losses_gd)

if __name__ == "__main__": #main function
    n = 100 #no.of datapoints
    m2=0.1 #initialising intercept with random value
    x, y = generate_data(n)
    plot_data(x, y)

    reg = train_model(x, y)

    plot_best_fit(x, y, reg)
    y_pred = reg.predict(x.reshape(-1, 1))
    mse, mse_list = compute_mse(y, y_pred)
    print("Mean Squared Error:", mse)
    plot_mse(x, mse_list)

    best_m1, best_m2 = reg.coef_[0], reg.intercept_
    #best slope determines how much y changes for increase in each x units
    print("Best slope:", best_m1, "Best intercept:", best_m2)

    plot_mse_vs_params(x, y, best_m1, best_m2, n)
    m1_range = np.linspace(0, 5, n)

    best_m1_brute = brute_force_scan(m2,x, y, m1_range)

    best_m1_gd, losses_gd = gradient_descent(m2,x, y)

    print("Efficiency:", efficiency(m1_range, losses_gd)) #efficiency

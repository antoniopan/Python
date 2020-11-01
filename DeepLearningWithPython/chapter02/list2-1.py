import matplotlib.pyplot as plt
import numpy as np


def regression():
    x = np.linspace(-1, 1, 100)
    signal = 2 + x + 2 * x ** 2
    noise = np.random.normal(0, 0.1, 100)
    y = signal + noise
    plt.plot(signal, 'b')
    plt.plot(y, 'g')
    plt.plot(noise, 'r')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["Without Noise", "With Noise", "Noise"], loc = 2)
    x_train = x[0:80]
    y_train = y[0:80]

    plt.figure()
    degree = 2
    X_train = np.column_stack([np.power(x_train, i) for i in range(0, degree)])
    model = np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(), X_train)), X_train.transpose()), y_train)
    plt.plot(x, y, 'g')
    plt.xlabel("x")
    plt.ylabel("y")
    predicted = np.dot(model, [np.power(x, i) for i in range(0, degree)])
    plt.plot(x, predicted, 'r')
    plt.legend(["Actual", "Predicted"], loc = 2)
    train_rmse1 = np.sqrt(np.sum(np.dot(y[0:80] - predicted[0:80], y_train - predicted[0:80])))
    test_rmse1 = np.sqrt(np.sum(np.dot(y[80:] - predicted[80:], y[80:] - predicted[80:])))
    print("Train RMSE (Degree = 1", train_rmse1)
    print("Test RMSE (Degree = 1)", test_rmse1)

    plt.figure()
    degree = 9
    X_train = np.column_stack([np.power(x_train, i) for i in range(0, degree)])
    model = np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(), X_train)), X_train.transpose()), y_train)
    plt.plot(x, y, 'g')
    plt.xlabel("x")
    plt.ylabel("y")
    predicted = np.dot(model, [np.power(x, i) for i in range(0, degree)])
    plt.plot(x, predicted, 'r')
    plt.legend(["Actual", "Predicted"], loc=2)
    train_rmse2 = np.sqrt(np.sum(np.dot(y[0:80] - predicted[0:80], y_train - predicted[0:80])))
    test_rmse2 = np.sqrt(np.sum(np.dot(y[80:] - predicted[80:], y[80:] - predicted[80:])))
    print("Train RMSE (Degree = 2", train_rmse2)
    print("Test RMSE (Degree = 2)", test_rmse2)

    plt.show()

    return

if __name__ == '__main__':
    regression()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


class LinearSupportVectorMachine():
    
    def __init__(self) -> None:
        self.__w = None
        self.__b = None
        

    @property
    def weights(self):
        return self.__w / np.linalg.norm(self.__w)

    @property
    def bias(self):
        return self.__b / np.linalg.norm(self.__w)
    

    def fit(self, X, y, iterations=100, learning_rate=1e-2, lambda_=1, C=100):
        m, n = X.shape

        self.__w = np.random.randn(n, 1)
        self.__b = np.random.randn(1, 1)

        step = iterations // 10

        for iteration in range(iterations):
            dJ_dw = 2 * lambda_ * self.__w - C * np.transpose(np.sum((1 - y * (X @ self.__w + self.__b) > 0) * y * X, axis=0)).reshape(n, 1) 
            dJ_db = - C * np.sum((1 - y * (X @ self.__w + self.__b) > 0) * y)

            self.__w -= learning_rate * dJ_dw
            self.__b -= learning_rate * dJ_db

            cost_J = lambda_ * np.linalg.norm(self.__w) ** 2 + C * np.sum(np.maximum(0, 1 - y * (X @ self.__w + self.__b)))

            if (iteration + 1) % step == 0:
                print(f"Iteration {iteration + 1}: Cost: {cost_J}")

    def predict(self, X):
        return np.sign(X @ self.__w + self.__b)


if __name__ == '__main__':
    m = 100

    np.random.seed(1)

    class0 = np.random.multivariate_normal(np.array([0, 0]), 2.5 * np.eye(2), m // 2)
    class1 = np.random.multivariate_normal(np.array([5, 5]), 2.5 * np.eye(2), m // 2)

    y0 = -1 * np.ones((m // 2, 1)) 
    y1 = np.ones((m // 2, 1))

    data0 = np.concatenate([class0, y0], axis=1)
    data1 = np.concatenate([class1, y1], axis=1)

    dataset = np.concatenate([data0, data1])
    np.random.shuffle(dataset)

    X = dataset[:, :2]
    y = dataset[:, 2].reshape(-1, 1)

    model = LinearSupportVectorMachine()
    model.fit(X, y, iterations=10000, lambda_=1, C=100)


    decision_boundary_slope = - model.weights[0, 0] / model.weights[1, 0]
    decision_boundary_intercept = - model.bias[0, 0] / model.weights[1, 0]


    x_bound = np.array([-30, 30])
    y_bound = decision_boundary_slope * x_bound + decision_boundary_intercept

    print(model.weights)
    print(model.bias)

    model_lib = SVC(C=300, kernel='linear')
    model_lib.fit(X, y)

    print(model_lib.intercept_)
    print(model_lib.coef_)




    plt.scatter(dataset[:, 0], dataset[:, 1], s=100, edgecolors='black', c=dataset[:, 2])
    plt.plot(x_bound, y_bound, label="Decision boundary")

    plt.xlim(-10, 15)
    plt.ylim(-10, 15)
    plt.legend()

    plt.show()
import numpy as np


class LinearRegression():

    def __init__(self) -> None:
        self.__w = None
        self.__b = None

    @property
    def weights(self):
        return self.__w
    
    @property
    def bias(self):
        return self.__b
    

    def fit(self, X, y, iterations=100, learning_rate=1e-2, lambda_=1):
        m, n = X.shape

        self.__w = np.random.randn(n, 1)
        self.__b = np.random.randn(1, 1)

        for iteration in range(iterations):

            dJ_dw = 2 * np.transpose(np.mean((X @ self.__w + self.__b - y) * X, axis=0)).reshape(n, 1) + 2 * (lambda_ / m) * self.__w
            dJ_db = 2 * np.mean(X @ self.__w + self.__b - y)

            self.__w -= learning_rate * dJ_dw
            self.__b -= learning_rate * dJ_db


            cost_J = np.mean((X @ self.__w + self.__b - y) ** 2) + (lambda_ / m) * np.linalg.norm(self.__w) ** 2

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Cost: {cost_J}")

        
    def predict(self, X):
        return X @ self.__w + self.__b



if __name__=="__main__":
    X = np.random.randn(100, 10)
    
    true_weights = np.array([4.6, 2.4, -3.5, 1.4, 1.4, 0.65, 2.43, 0.325, 2.143, 3.111]).reshape(10, 1)

    true_bias = np.array([1.3434])

    y = X @ true_weights + true_bias 

    # temp = np.transpose(np.mean((X @ true_weights + true_bias - y) * X, axis=0)).reshape(10, 1)
    # print(temp.shape)
    model = LinearRegression()

    model.fit(X, y)
    
    weights = model.weights.reshape(10, )
    bias = model.bias.reshape(1)[0]

    for i in range(len(weights)):
        print(f"w{i + 1} = {weights[i]}")
    
    print(f"b: {bias}")


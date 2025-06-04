import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()

        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.m
        db = - 2 * np.sum(self.Y - Y_pred) / self.m
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    def predict(self, X):
        return X.dot(self.W) + self.b


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

x_train = train_df.drop(columns=['MedHouseVal'])
y_train = train_df['MedHouseVal']


def fit_and_plot(feature, color):
    model = LinearRegression(learning_rate=0.01, iterations=1000)
    X_feature = x_train[[feature]].values
    y = y_train.values
    model.fit(X_feature, y)
    predictions = model.predict(X_feature)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_feature, y, alpha=0.5, color=color)
    plt.plot(X_feature, predictions, color="black", linestyle='--')
    plt.xlabel(feature)
    plt.ylabel('MedHouseVal')
    plt.title(f'{feature} vs MedHouseVal')
    plt.show()


fit_and_plot('HouseAge', 'blue')
fit_and_plot('AveRooms', 'red')
fit_and_plot('AveBedrms', 'pink')
fit_and_plot('Population', 'orange')
fit_and_plot('AveOccup', 'yellow')
fit_and_plot('Latitude', 'purple')
fit_and_plot('Longitude', 'cyan')
fit_and_plot('MedInc', 'green')


X_full = x_train.values
y_full = y_train.values

final_model = LinearRegression(learning_rate=0.01, iterations=1000)
final_model.fit(X_full, y_full)


X_test = test_df.values
test_predictions = final_model.predict(X_test)


submission = pd.DataFrame({
    'Id': np.arange(len(test_predictions)),
    'target': test_predictions
})


submission.to_csv('submission.csv', index=False)

print(" Submission file 'submission.csv' created successfully.")

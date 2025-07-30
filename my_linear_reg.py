import numpy as np
class myLinearReg():
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):
        num = 0
        den = 0
        for i in range(X_train.shape[0]):
            num += (X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean())
            den += (X_train[i] - X_train.mean()) ** 2
        self.m = num / den
        self.b = y_train.mean() - self.m * X_train.mean()

    def predict(self, X_test):
        return self.m * X_test + self.b

    def metrics(self, y_test, y_pred):
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)

        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        r2_num = np.sum((y_test - y_pred) ** 2)
        r2_den = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (r2_num / r2_den)

        return {
            "MSE": round(mse, 3),
            "MAE": round(mae, 3),
            "R2 Score": round(r2, 3)
        }


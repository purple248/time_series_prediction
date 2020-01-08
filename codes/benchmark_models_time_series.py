import csv
import numpy as np
import time
import pandas as pd

class BasicModels:

    def __init__(self, x_train, x_test, y_train, y_test):

        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test

    def print_save_errors(self, true_y, predicted_y, model_name):
        from sklearn.metrics import r2_score
        from sklearn import metrics
        r2_square = r2_score(true_y, predicted_y)
        mse = metrics.mean_squared_error(true_y, predicted_y)
        mae = metrics.mean_absolute_error(true_y, predicted_y)
        rmse = np.sqrt(metrics.mean_squared_error(true_y, predicted_y))

        print("{} results".format(model_name))
        print(f"R^2: {r2_square}")
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}\n")

        import matplotlib.pyplot as plt
        # plt.plot(range(len(self.y_test)), self.y_test, lw=1.5, color='blue', label='original')
        # plt.plot(range(len(self.y_test)), predicted_y, lw=1.5, color='red', label='predicted')
        plt.plot(range(100), self.y_test[-100:], lw=1.5, color='blue', label='original')
        plt.plot(range(100), predicted_y[-100:], lw=1.5, color='red', label='predicted')
        plt.legend()
        plt.title(model_name)
        plt.show()

        file_path = "../results/benchmark_rez.csv"
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_name, r2_square, mse, mae, rmse])
        print("test rez saved")
        return



    def LinearRegression(self):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(self.x_train, self.y_train)

        print("linear regression results:")
        #print(list(zip(self.x_train.columns, model.coef_)))
        preds = model.predict(self.x_test)

        x_train_new = pd.DataFrame(self.x_train)
        x_train_new['bias'] = 1
        theta_hat = np.dot(np.dot(np.linalg.inv(np.dot(x_train_new.T, x_train_new)), x_train_new.T), self.y_train)
        print("theta_hat: {}" .format(theta_hat))

        self.print_save_errors(self.y_test, preds, "linear regression")

    def krr(self):
        from sklearn.kernel_ridge import KernelRidge
        model_lin = KernelRidge(alpha=1.0)
        model_lin.fit(self.x_train, self.y_train)
        preds = model_lin.predict(self.x_test)
        self.print_save_errors(self.y_test, preds, "KRR linear")

        model_rbf = KernelRidge(alpha=1.0, kernel='rbf')
        model_rbf.fit(self.x_train, self.y_train)
        preds = model_rbf.predict(self.x_test)
        self.print_save_errors(self.y_test, preds, "KRR rbf")

        model_poly = KernelRidge(alpha=1.0, kernel='poly')
        model_poly.fit(self.x_train, self.y_train)
        preds = model_poly.predict(self.x_test)
        self.print_save_errors(self.y_test, preds, "KRR polynomial")

    def svr(self):
        from sklearn.svm import SVR
        model_lin = SVR(kernel='linear' , epsilon=0.01) #gamma='scale' - less good results
        model_rbf = SVR(kernel='rbf', epsilon=0.01, gamma= 0.19) #after checking for the best gamma
        model_poly = SVR(kernel='poly', epsilon=0.01, degree = 3) #Degree of the polynomial kernel function - defult 3

        a = time.time()
        model_lin.fit(self.x_train, self.y_train)
        b = time.time()
        preds_lin = model_lin.predict(self.x_test)
        c = time.time()
        self.print_save_errors(self.y_test, preds_lin, "SVR - linear")
        print(model_lin.support_vectors_.shape)
        print("fit train time {}".format(b - a))
        print("test preds time {}".format(c - b))

        model_rbf.fit(self.x_train, self.y_train)
        preds_rbf = model_rbf.predict(self.x_test)
        self.print_save_errors(self.y_test, preds_rbf, "SVR - rbf")
        print(model_rbf.support_vectors_.shape)

        model_poly.fit(self.x_train, self.y_train)
        preds_poly = model_poly.predict(self.x_test)
        self.print_save_errors(self.y_test, preds_poly, "SVR - polynomial")
        print(model_poly.support_vectors_.shape)

    def random_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        model = RandomForestRegressor(max_depth=4, random_state=0, n_estimators=100)
        model.fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)
        self.print_save_errors(self.y_test, preds, "random forest")

    # def autoregression(self): #TODO add autoregression
    #     from statsmodels.tsa.ar_model import AR



if __name__ == '__main__':
    from data_handler_for_benchmarks import DataHandler
    dh = DataHandler()
    x_train, x_test, y_train, y_test = dh.normalization()

    bm = BasicModels(x_train, x_test, y_train, y_test)

    data_name = "20k_2_sine_signal_no_noise"
    file_path = "../results/benchmark_rez.csv"
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow ([data_name])
        writer.writerow(["model_name", "r2_square", "mse", "mae", "rmse"])

    bm.LinearRegression()
    bm.svr()
    bm.krr()
    bm.random_forest()

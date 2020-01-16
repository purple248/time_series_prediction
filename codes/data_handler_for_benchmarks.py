import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle


class DataHandler:

    def __init__(self):
        with open('../data/2_sine_signal.pickel', 'rb') as handle:
            self.generated_data = pickle.load(handle)

        # another optional data:
        # with open('../data/triple_sine_signal.pickel', 'rb') as handle:
        #     self.generated_data = pickle.load(handle)

        self.df = pd.DataFrame(data=self.generated_data, index=None, columns=['x'])
        self.adding_history()
        self.train_test_split()


    def check_autocorr(self):
        #checking for autocorralation:
        autocorr_data = self.generated_data
        x = np.log(autocorr_data[1:] / autocorr_data[:-1])
        plt.acorr(x, maxlags=9)
        plt.xlabel('lag')
        plt.ylabel('autocorralation')
        plt.show()


    def adding_history(self):
        #adding 3 days historical raw_signal:
        self.df['x-1'] = np.roll(self.df['x'], 1)
        self.df['x-2'] = np.roll(self.df['x'], 2)
        self.df['x-3'] = np.roll(self.df['x'], 3)
        self.df['y'] = np.roll(self.df['x'], -1)

        #trimm 0-2 rows with missing history:
        self.df = self.df.iloc[3:]

        ## to save the data with history:
        # self.df.to_pickle("../data/generated_data_with_history_3days")

    def check_distribution(self):
        # Visualizing the distribution of features to see if is normal distribution:
        import seaborn as sns
        sns.distplot(self.df['x'])
        plt.show()
        #after log:
        try:
            sns.distplot(np.log(self.df['x']))
        except:
            print("error in distplot, printing histogram instead")
            plt.hist(np.log(self.df['x']))
        plt.show()


    def train_test_split(self, train_test_ratio = 0.8):
        # because it is a time series data - the split is on earlier period (train) and last period (test)
        split_point = int(len(self.df) * train_test_ratio)
        x_data = np.array(self.df[['x','x-1','x-2','x-3']])
        y_data = np.array(self.df['y'])
        self.x_train = x_data[:split_point,:]
        self.y_train = y_data[:split_point]

        self.x_test = x_data[split_point:, :]
        self.y_test = y_data[split_point:]


    def normalization(self):
        #scaler:
        x_mean_train = np.mean(self.x_train, axis=0)
        x_std_train = np.std(self.x_train, axis=0)
        # normalization:
        self.x_train = np.dot((self.x_train - x_mean_train),np.diag(1/x_std_train))
        self.x_test = np.dot((self.x_test - x_mean_train),np.diag(1/x_std_train))

        return self.x_train, self.x_test, self.y_train, self.y_test






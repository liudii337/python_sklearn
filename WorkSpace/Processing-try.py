import time
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

data_train = np.load('Data-Train.npy')
data_test = np.load('Data-Test.npy')

X_train0 = data_train[:, :-1]
y_train0 = data_train[:, -1:]
X_test0 = data_test[:, :-1]
y_test0 = data_test[:, -1:]

# scio.savemat('data.mat', {
#     'X_train': X_train0,
#     'y_train': y_train0,
#     'X_test': X_test0,
#     'y_test':y_test0,
# })

ss_X = MinMaxScaler()
ss_y = MinMaxScaler()

X_trans = ss_X.fit_transform(X_train0)
y_trans = ss_y.fit_transform(y_train0)

#测试集不用打乱
X_test = ss_X.transform(X_test0)
y_test = ss_y.transform(y_test0)
#将训练集随机打乱
X, y = shuffle(X_trans, y_trans, random_state=0)

#线性回归
from sklearn import linear_model
linear_reg = linear_model.LinearRegression()

#树回归
from sklearn import tree
tree_reg = tree.DecisionTreeRegressor()

#SVM回归
from sklearn import svm
svr = svm.SVR(C=3,gamma=1.25,kernel='rbf')

#KNN
from sklearn import neighbors
knn = neighbors.KNeighborsRegressor()

#MLP
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=18,alpha=1e-5,learning_rate_init=0.01,max_iter=2000,solver='adam')

#集成方法
#随机森林
from sklearn import ensemble
rf = ensemble.RandomForestRegressor(n_estimators=400)  #这里使用20个决策树

#AdaBoost
ada = ensemble.AdaBoostRegressor(n_estimators=50)

#GBRT
gbrt = ensemble.GradientBoostingRegressor(n_estimators=100)

methods = [{
    # "linear": linear_reg,
    # "tree": tree_reg,
    # "svr": svr,
    # "KNN": knn,
    "MLP": mlp,
    # "RandomForest": rf,
    # "Adaboost": ada,
    # "GBRT": gbrt
}]

for i in methods:
    for name, estimator in i.items():
        train_time = []
        test_time = []

        t0 = time.time()
        estimator.fit(X, y.ravel())
        train_time_this = time.time() - t0
        train_time.append(train_time_this)

        t0 = time.time()
        result = estimator.predict(X_test)
        test_time_this = time.time() - t0
        test_time.append(test_time_this)

        score = estimator.score(X_test, y_test)
        mse = mean_squared_error(
            ss_y.inverse_transform(y_test), ss_y.inverse_transform(result.reshape(-1, 1)))
        
        # plt.figure()
        # plt.plot(np.arange(0,48), y_test[11*48:11*48+48],'go-',label='true value')
        # plt.plot(np.arange(0,48),result[11*48:11*48+48],'ro-',label='predict value')
        # plt.title('%s score: %f'%(name,score))
        # plt.legend()
        
        print("%s (mse: %3f score: %.3f fit: %.3fs, predict: %.3fs)" %
              (name, mse, score, train_time_this, test_time_this))
        error_y = (ss_y.inverse_transform(result.reshape(-1, 1)) - ss_y.inverse_transform(y_test))/ss_y.inverse_transform(y_test)
        abs_error_y = np.abs(error_y)
        print("\terror max:%.3f mean:%.3f" %(abs_error_y.max(),abs_error_y.mean()))

        predict=(ss_y.inverse_transform(result.reshape(-1, 1))).reshape(31,48)
        np.save('out-single.npy', predict)

plt.show()
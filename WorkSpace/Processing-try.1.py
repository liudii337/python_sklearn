import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

maindata = np.load('AllData.npy')

X = maindata[:, :-1]
y = maindata[:, -1:]

ss_X = MinMaxScaler()
ss_y = MinMaxScaler()

X_trans = ss_X.fit_transform(X)
y_trans = ss_y.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X_trans, y_trans, random_state=22, test_size=0.25)
'''
print ('The max target value is: ', np.max(y))
print ('The min target value is: ', np.min(y))
print ('The average terget value is: ', np.mean(y))
'''
score_each = []


def try_different_method(clf):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    result = clf.predict(X_test)
    plt.figure()
    plt.plot(np.arange(0, 48), y_test[0:48], 'go-', label='true value')
    plt.plot(np.arange(0, 48), result[0:48], 'ro-', label='predict value')
    plt.title('%s score: %f' % (clf, score))
    score_each.append(score)
    plt.legend()


'''
print ('The value of default measurement of LinearRegression is: ', lr.score(X_test, y_test))

print ('The value of R-squared of LinearRegression is: ', r2_score(y_test, lr_y_predict))
print ('The value of mean squared error of LinearRegression is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
print ('The value of mean absolute error of LinearRegression is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
'''

#线性回归
from sklearn import linear_model
linear_reg = linear_model.LinearRegression()

#树回归
from sklearn import tree
tree_reg = tree.DecisionTreeRegressor()

#SVM回归
from sklearn import svm
svr = svm.SVR(C=2.5, gamma=1, kernel='rbf')

#KNN
from sklearn import neighbors
knn = neighbors.KNeighborsRegressor(
    n_neighbors=8, weights='distance', algorithm='ball_tree')

#MLP
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(
    hidden_layer_sizes=(200, 96),
    solver='lbfgs',
    activation='relu',
    alpha=0.1,
    learning_rate_init=0.01)

#110 120
#集成方法
#随机森林
from sklearn import ensemble
rf = ensemble.RandomForestRegressor(n_estimators=400)  #这里使用20个决策树

#AdaBoost
ada = ensemble.AdaBoostRegressor(n_estimators=50)

#GBRT
gbrt = ensemble.GradientBoostingRegressor(n_estimators=100)

methods = [{
    "linear": linear_reg,
    "tree": tree_reg,
    "MLP": mlp,
    "svr": svr,
    "KNN": knn,
    "RandomForest": rf,
    "Adaboost": ada,
    "GBRT": gbrt
}]

for i in methods:
    for name, estimator in i.items():
        train_time = []
        test_time = []

        t0 = time.time()
        estimator.fit(X_train, y_train.ravel())
        train_time_this = time.time() - t0
        train_time.append(train_time_this)

        t0 = time.time()
        result = estimator.predict(X_test)
        test_time_this = time.time() - t0
        test_time.append(test_time_this)

        score = estimator.score(X_test, y_test)
        score_each.append(score)
        mse = mean_squared_error(
            ss_y.inverse_transform(y_test),
            ss_y.inverse_transform(result.reshape(-1, 1)))
        '''
        plt.figure()
        plt.plot(np.arange(0,48), y_test[0:48],'go-',label='true value')
        plt.plot(np.arange(0,48),result[0:48],'ro-',label='predict value')
        plt.title('%s score: %f'%(name,score))
        plt.legend()
        '''
        print("%s (mse: %3f score: %.3f fit: %.3fs, predict: %.3fs)" %
              (name, mse, score, train_time_this, test_time_this))

plt.show()
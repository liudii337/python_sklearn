import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR

def plotday(start, result, y_test):
    plt.figure()
    plt.plot(
        np.arange(0, 48),
        y_test[start * 48:start * 48 + 48],
        'go-',
        label='true value')
    plt.plot(
        np.arange(0, 48),
        result[start * 48:start * 48 + 48],
        'ro-',
        label='predict value')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    data_train = np.load('Data-Train.npy')
    data_test = np.load('Data-Test.npy')

    X_train0 = data_train[:, :-1]
    y_train0 = data_train[:, -1:]
    X_test0 = data_test[:, :-1]
    y_test0 = data_test[:, -1:]

    ss_X = MinMaxScaler()
    ss_y = MinMaxScaler()

    X_trans = ss_X.fit_transform(X_train0)
    y_trans = ss_y.fit_transform(y_train0)

    #测试集不用打乱
    X_test = ss_X.transform(X_test0)
    y_test = ss_y.transform(y_test0)
    #将训练集随机打乱
    X, y = shuffle(X_trans, y_trans, random_state=0)
    '''
    train_sizes,train_loss,test_loss=learning_curve(
        SVR(), X , y.ravel() , cv=5 , scoring='neg_mean_squared_error',
        train_sizes=[0.1,0.25,0.5,0.75,1])

    train_loss_mean=-np.mean(train_loss,axis=1)
    test_loss_mean=-np.mean(test_loss,axis=1)

    plt.plot(train_sizes,train_loss_mean,'o-',color='r',
         label="Training")
    plt.plot(train_sizes,test_loss_mean,'o-',color='g',
         label="Cross-validation")

    plt.xlabel("Training examples")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()

    '''
    param_grid = {
        'kernel': ['rbf'],
        'C': [1.5, 2, 2.5, 3],
        'gamma': [0.75, 1, 1.1, 1.25]
    }
    '''
    
    'kernel':('linear', 'rbf', 'poly', 'sigmoid'),
    'C':[1.5,2,2.5,3],
    'gamma':[0.75,1,1.1,1.25]
    '''
    grid = GridSearchCV(SVR(), cv=4, param_grid=param_grid, n_jobs=2)

    t0 = time.time()
    grid.fit(X, y.ravel())
    fit_time = time.time() - t0

    t0 = time.time()
    result = grid.predict(X_test)
    predict_time = time.time() - t0
    score = grid.score(X_test, y_test)

    print("SVR:")
    print("fit time: %3fs" %(fit_time))
    print("predict time: %3fs" %(predict_time))
    print("test score: %3f" % (score))
    print("best parameter: %s " % (grid.best_params_))
    print("best score: %s " % (grid.best_score_))
    print(grid.grid_scores_)
    
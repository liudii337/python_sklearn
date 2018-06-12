import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

maindata = np.load('maindata1.npy')

X=maindata[:,:-1]
y=maindata[:,-1:]

ss_X = StandardScaler()
ss_y = StandardScaler()

X_trans=ss_X.fit_transform(X)
y_trans=ss_y.fit_transform(y.reshape(-1,1))

X, y = shuffle(X_trans, y_trans, random_state=0)

'''
train_sizes,train_loss,test_loss=learning_curve(
    RandomForestRegressor(), X , y,cv=5 ,scoring='neg_mean_squared_error',
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
param_grid = {"n_estimators": [100,200,300,400]}

grid = GridSearchCV(RandomForestRegressor(), cv=4, param_grid=param_grid)
  
t0 = time.time()
grid.fit(X,y.ravel())
fit_time=time.time() - t0

print("RandomForest:")
print("fit time: %3fs" % (fit_time))
print("best parameter: %s " % (grid.best_params_))
print("best score: %s " % (grid.best_score_))
print(grid.grid_scores_)

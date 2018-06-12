import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


maindata = np.load('maindata1.npy')

X=maindata[:,:-1]
y=maindata[:,-1:]

ss_X = StandardScaler()
ss_y = StandardScaler()

X_trans=ss_X.fit_transform(X)
y_trans=ss_y.fit_transform(y.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X_trans, y_trans, random_state = 33, test_size = 0.25)
'''
print ('The max target value is: ', np.max(y))
print ('The min target value is: ', np.min(y))
print ('The average terget value is: ', np.mean(y))
'''

lr = LinearRegression()
svr = SVR(gamma=0.8)

lr.fit(X_train, y_train)
svr.fit(X_train, y_train.ravel())

lr_y_predict = lr.predict(X_test)
svr_y_predict = svr.predict(X_test)
svr_y=svr.predict(X_trans)
'''
print ('The value of default measurement of LinearRegression is: ', lr.score(X_test, y_test))

print ('The value of R-squared of LinearRegression is: ', r2_score(y_test, lr_y_predict))
print ('The value of mean squared error of LinearRegression is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
print ('The value of mean absolute error of LinearRegression is: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
'''
print ('The value of mean squared error of SVR is: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(svr_y_predict)))


y_test=ss_y.inverse_transform(y_test)
lr_y_predict=ss_y.inverse_transform(lr_y_predict)
svr_y_predict=ss_y.inverse_transform(svr_y_predict)
svr_y=ss_y.inverse_transform(svr_y)

X_plot=np.linspace(0.5,24,48)
'''
from sklearn.learning_curve import learning_curve
train_sizes,train_loss,test_loss=learning_curve(
    SVR(gamma=0.01),X_trans,y_trans,cv=10,scoring='neg_mean_squared_error',
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
'''
'''
from sklearn.learning_curve import validation_curve
para_range=np.logspace(-6,1,8)

train_loss,test_loss=validation_curve(
    SVR(),X_trans,y_trans,param_name='gamma',param_range=para_range,cv=10,scoring='neg_mean_squared_error',
    )

train_loss_mean=-np.mean(train_loss,axis=1)
test_loss_mean=-np.mean(test_loss,axis=1)

plt.plot(para_range,train_loss_mean,'o-',color='r',
         label="Training")
plt.plot(para_range,test_loss_mean,'o-',color='g',
         label="Cross-validation")

plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
'''
'''
plt.figure()
lw=1
plt.subplot(2,1,1)
plt.plot(X_plot,y_test[0:48],'go-',lw=lw,label='Test')
plt.plot(X_plot,lr_y_predict[0:48],'ro-',lw=lw,label='LR Predict')
plt.ylabel('Load')
plt.xlim(0,24)
plt.title('One Day Load')
plt.legend()

plt.subplot(2,1,2)
plt.plot(X_plot,y_test[0:48],'go-',lw=lw,label='Test')
plt.plot(X_plot,svr_y_predict[0:48],'bo-',lw=lw,label='SVR Predict')

plt.xlabel('Time')
plt.ylabel('Load')
plt.xlim(0,24)
plt.legend()
'''

plt.figure()
lw=1
plt.plot(X_plot,y[0:48],'go-',lw=lw,label='Real')
plt.plot(X_plot,svr_y[0:48],'bo-',lw=lw,label='SVR Predict')

plt.xlabel('Time')
plt.ylabel('Load')
plt.title('One Day Load')
plt.xlim(0,24)
plt.legend()

plt.show()
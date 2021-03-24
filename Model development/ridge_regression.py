from preprocessing import x_train,x_test,y_train,y_test
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from time import time

#linear regression
ridge = Ridge()
t0 = time()
ridge.fit(x_train,y_train)
print("ridge tarining time:", round(time()-t0, 3), "s")
y_predict = ridge.predict(x_test)

#calculate MSE
MSE = mean_squared_error(y_predict,y_test)

print("Ridge Regression RMSE is:",np.sqrt(MSE))

#calculate R-squared
r2 = r2_score(y_test, y_predict)

print("Ridge  r2 is:",r2)

#plot the predicted v. actual

y_plot = list(y_test)

plt.plot(y_predict)
plt.plot(y_plot)
plt.show()
from preprocessing import x_train,x_test,y_train,y_test
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from time import time

#linear regression
dt = DecisionTreeRegressor(max_depth=5, random_state=20)

t0 = time()
dt.fit(x_train,y_train)
print("DT tarining time:", round(time()-t0, 3), "s")
y_predict = dt.predict(x_test)

#calculate MSE
MSE = mean_squared_error(y_predict,y_test)

print("Decision Tree RMSE is:",np.sqrt(MSE))

#calculate R-squared
r2 = r2_score(y_test, y_predict)

print("Decision Tree Regression  r2 is:",r2)

#plot the predicted v. actual

y_plot = list(y_test)

plt.plot(y_predict)
plt.plot(y_plot)
plt.show()
from preprocessing import x_train,x_test,y_train,y_test
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from time import time

#linear regression
lm = LinearRegression()
t0 = time()
lm.fit(x_train,y_train)
print("LM training time:", round(time()-t0, 3), "s")

y_predict = lm.predict(x_test)

#calculate MSE
MSE = mean_squared_error(y_predict,y_test)

print("Multivariate Regression RMSE is:",np.sqrt(MSE))

#calculate R-squared
r2 = r2_score(y_test, y_predict)

print("Multivariate Regression  r2 is:",r2)

PMSE = MSE / (sum(y_test) / len(y_test))

print("Percentage MSE is:", np.sqrt(PMSE))

#plot the predicted v. actual

y_plot = list(y_test)

plt.plot(y_predict)
plt.plot(y_plot)
plt.show()


plt.plot(y_plot,y_predict,'o')
plt.show()
from preprocessing import x_train,x_test,y_train,y_test, x, y, x_standard
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from time import time
import pandas as pd
import statsmodels.api as sm
from scipy import stats



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

coeff_df = pd.DataFrame(lm.coef_, x.columns, columns=['Coefficient'])
coeff = coeff_df.sort_values(by='Coefficient', ascending=False)
print(coeff)

X2 = sm.add_constant(x_train)
Y2 = list(y_train)
mreg2 = sm.OLS(Y2, X2).fit()
print(mreg2.summary())

ytest = pd.DataFrame(y_test)
linear = pd.DataFrame(y_predict)

ytest.to_csv("D:/Academic/Alta Phd/E-ML models/Results/y_test.csv")
linear.to_csv("D:/Academic/Alta Phd/E-ML models/Results/linear.csv")
coeff.to_csv("D:/Academic/Alta Phd/E-ML models/Results/linear_coeff.csv")





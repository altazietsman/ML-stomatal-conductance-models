from preprocessing import x_train,x_test,y_train,y_test,x,y
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from time import time
import pandas as pd

#linear regression
dt = DecisionTreeRegressor(random_state=20)

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

PMSE = MSE / (sum(y_test) / len(y_test))

print("Percentage MSE is:", np.sqrt(PMSE))

#plot the predicted v. actual

y_plot = list(y_test)

plt.plot(y_predict)
plt.plot(y_plot)
plt.show()

feature_importance = list(dt.feature_importances_)
feature_names = list(x.columns)

features_df = pd.DataFrame()
features_df['feature'] = feature_names
features_df['importance'] = feature_importance

# re-organize it features
features_df = features_df.sort_values('importance', ascending=False)

print(features_df.round(1))

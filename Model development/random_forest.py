from preprocessing import x_train, x_test, y_train, y_test, df1, x, y
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from time import time
from sklearn.metrics import confusion_matrix


# Random Forest
rf = RandomForestRegressor(random_state=20, bootstrap=True)
t0 = time()
rf.fit(x_train, y_train)
print("RF training time:", round(time() - t0, 3), "s")

y_predict = rf.predict(x_test)

# calculate MSE
MSE = mean_squared_error(y_predict, y_test)

print("Random Forest RMSE is:", np.sqrt(MSE))

PMSE = MSE / (sum(y_test) / len(y_test))

print("Percentage MSE is:", np.sqrt(PMSE))

# calculate R-squared
r2 = r2_score(y_test, y_predict)

print("Random Forest Regression  r2 is:", r2)

# plot the predicted v. actual

y_plot = list(y_test)

plt.plot(y_predict)
plt.plot(y_plot)
plt.show()

# Check importance of features

feature_importance = list(rf.feature_importances_)
feature_names = list(x.columns)

features_df = pd.DataFrame()
features_df['feature'] = feature_names
features_df['importance'] = feature_importance

# re-organize it features
features_df = features_df.sort_values('importance', ascending=False)

print(features_df.round(1))

# Bootstrapping

rf_bag = BaggingRegressor(base_estimator=rf)
t1 = time()
rf_bag.fit(x_train, y_train)
print("RF bootstrap tarining time:", round(time() - t1, 3), "s")
y_rf_bag_pred = rf_bag.predict(x_test)

MSE_ref_bag = mean_squared_error(y_rf_bag_pred, y_test)

print("Random Forest bootstrap RMSE is:", np.sqrt(MSE_ref_bag))

PMSE_bag = MSE_ref_bag / (sum(y_test) / len(y_test))

print("Percentage bootstrap MSE is:", np.sqrt(PMSE_bag))

# calculate R-squared
r2_bag = r2_score(y_test, y_rf_bag_pred)

print("Random Forest bootstrap  r2 is:", r2_bag)


plt.plot(y_rf_bag_pred, y_test,'o')
plt.show()

# Boosting

rf_bst = AdaBoostRegressor(base_estimator=rf)

t2 = time()
rf_bst.fit(x_train, y_train)
print("RF boosting training time:", round(time() - t2, 3), "s")

y_rf_bst_pred = rf_bst.predict(x_test)

MSE_ref_bst = mean_squared_error(y_rf_bst_pred, y_test)

print("Random Forest boost RMSE is:", np.sqrt(MSE_ref_bst))

PMSE_bst = MSE_ref_bst / (sum(y_test) / len(y_test))

print("Percentage boost MSE is:", np.sqrt(PMSE_bst))

# calculate R-squared
r2_bst = r2_score(y_test, y_rf_bst_pred)

print("Random Forest boost  r2 is:", r2_bst)

y_plot = list(y_test)

plt.plot(y_rf_bst_pred)
plt.plot(y_plot)
plt.show()


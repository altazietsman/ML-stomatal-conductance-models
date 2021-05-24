#test affect of training data size
from preprocessing import x_train20,y_train20, x_test20, y_test20
from preprocessing import x_train30,y_train30, x_test30, y_test30
from preprocessing import x_train40,y_train40, x_test40, y_test40
from preprocessing import x_train50,y_train50, x_test50, y_test50
from preprocessing import x_train60,y_train60, x_test60, y_test60
from preprocessing import x_train70,y_train70, x_test70, y_test70
from preprocessing import x_train80,y_train80, x_test80, y_test80
from preprocessing import x_train90,y_train90, x_test90, y_test90
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt

# Random Forest 20
rf20 = RandomForestRegressor(random_state=20, bootstrap=True)
rf20.fit(x_train20, y_train20)

y_predict20 = rf20.predict(x_test20)

RF20 = pd.DataFrame(y_predict20)

RF20.to_csv("D:/Academic/Alta Phd/E-ML models/Results/RF20.csv")

# calculate MSE
MSE20 = mean_squared_error(y_predict20, y_test20)

print("Random Forest RMSE 20 is:", np.sqrt(MSE20))

RMSE20 = MSE20 / (sum(y_test20) / len(y_test20))

print("Percentage MSE 20 is:", np.sqrt(RMSE20))

# calculate R-squared
r220 = r2_score(y_test20, y_predict20)

print("Random Forest Regression 20 r2 is:", r220)

# Random Forest 30
rf30 = RandomForestRegressor(random_state=20, bootstrap=True)
rf30.fit(x_train30, y_train30)

y_predict30 = rf30.predict(x_test30)

RF30 = pd.DataFrame(y_predict30)

RF30.to_csv("D:/Academic/Alta Phd/E-ML models/Results/RF30.csv")

# calculate MSE
MSE30 = mean_squared_error(y_predict30, y_test30)

print("Random Forest RMSE 30 is:", np.sqrt(MSE30))

RMSE30 = MSE30 / (sum(y_test30) / len(y_test30))

print("Percentage MSE 30 is:", np.sqrt(RMSE30))

# calculate R-squared
r230 = r2_score(y_test30, y_predict30)

print("Random Forest Regression 30 r2 is:", r230)



# Random Forest 40
rf40 = RandomForestRegressor(random_state=20, bootstrap=True)
rf40.fit(x_train40, y_train40)

y_predict40 = rf40.predict(x_test40)

RF40 = pd.DataFrame(y_predict40)

RF40.to_csv("D:/Academic/Alta Phd/E-ML models/Results/RF40.csv")

# calculate MSE
MSE40 = mean_squared_error(y_predict40, y_test40)

print("Random Forest RMSE 40 is:", np.sqrt(MSE40))

RMSE40 = MSE40 / (sum(y_test40) / len(y_test40))

print("Percentage MSE 40 is:", np.sqrt(RMSE40))

# calculate R-squared
r240 = r2_score(y_test40, y_predict40)

print("Random Forest Regression 40 r2 is:", r240)

# Random Forest 50
rf50 = RandomForestRegressor(random_state=20, bootstrap=True)
rf50.fit(x_train50, y_train50)

y_predict50 = rf50.predict(x_test50)

RF50 = pd.DataFrame(y_predict50)

RF50.to_csv("D:/Academic/Alta Phd/E-ML models/Results/RF50.csv")

# calculate MSE
MSE50 = mean_squared_error(y_predict50, y_test50)

print("Random Forest RMSE 50 is:", np.sqrt(MSE50))

RMSE50 = MSE50 / (sum(y_test50) / len(y_test50))

print("Percentage MSE 50 is:", np.sqrt(RMSE50))

# calculate R-squared
r250 = r2_score(y_test50, y_predict50)

print("Random Forest Regression 50 r2 is:", r250)

# Random Forest 60
rf60 = RandomForestRegressor(random_state=20, bootstrap=True)
rf60.fit(x_train60, y_train60)

y_predict60 = rf60.predict(x_test60)

RF60 = pd.DataFrame(y_predict60)

RF60.to_csv("D:/Academic/Alta Phd/E-ML models/Results/RF60.csv")

# calculate MSE
MSE60 = mean_squared_error(y_predict60, y_test60)

print("Random Forest RMSE 60 is:", np.sqrt(MSE60))

RMSE60 = MSE60 / (sum(y_test60) / len(y_test60))

print("Percentage MSE 60 is:", np.sqrt(RMSE60))

# calculate R-squared
r260 = r2_score(y_test60, y_predict60)

print("Random Forest Regression 60 r2 is:", r260)

# Random Forest 70
rf70 = RandomForestRegressor(random_state=20, bootstrap=True)
rf70.fit(x_train70, y_train70)

y_predict70 = rf70.predict(x_test70)

RF70 = pd.DataFrame(y_predict70)

RF70.to_csv("D:/Academic/Alta Phd/E-ML models/Results/RF70.csv")

# calculate MSE
MSE70 = mean_squared_error(y_predict70, y_test70)

print("Random Forest RMSE 70 is:", np.sqrt(MSE70))

RMSE70 = MSE70 / (sum(y_test70) / len(y_test70))

print("Percentage MSE 70 is:", np.sqrt(RMSE70))

# calculate R-squared
r270 = r2_score(y_test70, y_predict70)

print("Random Forest Regression 70 r2 is:", r270)

# Random Forest 80
rf80 = RandomForestRegressor(random_state=20, bootstrap=True)
rf80.fit(x_train80, y_train80)

y_predict80 = rf80.predict(x_test80)

RF80 = pd.DataFrame(y_predict80)

RF80.to_csv("D:/Academic/Alta Phd/E-ML models/Results/RF80.csv")

# calculate MSE
MSE80 = mean_squared_error(y_predict80, y_test80)

print("Random Forest RMSE 80 is:", np.sqrt(MSE80))

RMSE80 = MSE80 / (sum(y_test80) / len(y_test80))

print("Percentage MSE 80 is:", np.sqrt(RMSE80))

# calculate R-squared
r280 = r2_score(y_test80, y_predict80)

print("Random Forest Regression 80 r2 is:", r280)

# Random Forest 90
rf90 = RandomForestRegressor(random_state=20, bootstrap=True)
rf90.fit(x_train90, y_train90)

y_predict90 = rf90.predict(x_test90)

RF90 = pd.DataFrame(y_predict90)

RF90.to_csv("D:/Academic/Alta Phd/E-ML models/Results/RF90.csv")

# calculate MSE
MSE90 = mean_squared_error(y_predict90, y_test90)

print("Random Forest RMSE 90 is:", np.sqrt(MSE90))

RMSE90 = MSE90 / (sum(y_test90) / len(y_test90))

print("Percentage MSE 90 is:", np.sqrt(RMSE90))

# calculate R-squared
r290 = r2_score(y_test90, y_predict90)

print("Random Forest Regression 90 r2 is:", r290)

#plot R-squared

split = ['20','30','40','50','60','70','80','90']
size = [len(x_train20),len(x_train30),len(x_train40),len(x_train50),
       len(x_train60),len(x_train70),len(x_train80),len(x_train90)]
values = [r220,r230,r240,r250,r260,r270,r280,r290]

df = pd.DataFrame()
df['split'] = split
df['R-sqaured'] = values
df['size'] = size

df.to_csv("D:/Academic/Alta Phd/E-ML models/Results/train_size.csv")


plt.scatter(df['size'],df['R-sqaured'])
plt.xlabel('Sample size of training data set')
plt.ylabel('R-sqaured')
plt.show()
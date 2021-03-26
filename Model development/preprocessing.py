import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from seaborn import pairplot
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np

df_org = pd.read_excel('D:/Academic/Alta Phd/E-ML models/Input Data/df.xlsx')
df_anddereg = pd.read_csv('D:/Academic/Alta Phd/E-ML models/Input Data/AllData_EcologyLetters_Figshare_v1_318.csv')

df2 = df_anddereg[['Species','VPD','Tair','Patm','PARin','SWC','Cond']]
df2 = df2.replace(-9999.0, np.nan)

df_org['PARin'] = df_org['solar']*0.45*4.57

df1 = df_org[['Species','VPD','Tair','Patm','PARin','SWC','Cond']]

df_combined = df1.append(df2)


#create dummy variable for species column
df = pd.get_dummies(df_combined, drop_first=True)

#check for multo-collinearty and linearity

#pairplot(df_combined)
#plt.show()

#remove outliers (use z-scores)

df['z_score'] = stats.zscore(df['Cond'])

df.loc[df['z_score'].abs()<=3]

#split data into x and y and remove data
y = df['Cond']
x = df.drop(['Cond','z_score'], axis = 1)

#scale data

scaler = MinMaxScaler()

#scale X variables
x_scaled = scaler.fit_transform(x)

#Change scaled variables into dataframe
x_standard = pd.DataFrame(x_scaled, columns = x.columns)

x_standard = x_standard.fillna(x_standard.mean())


#create train-test split
x_train, x_test, y_train, y_test = (train_test_split(x_standard,y, test_size = 0.20, random_state = 20))


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

#convert LWP to SWC
#the soil parameters for the van genuchten curves for loamy sand was used
df_anddereg = df_anddereg.replace(-9999.0, np.nan)

n = 1.56
m = 1-1/n
alpha = 0.036
df_anddereg['SWC_new'] = 1/((1+(-1*(df_anddereg['LWPpredawn'])/alpha)**n)**m)
df_anddereg.loc[df_anddereg['SWC'].isnull(),'SWC'] = df_anddereg['SWC_new']

df2 = df_anddereg[['Species','VPD','PARin','SWC','Cond']]


df_org['PARin'] = df_org['solar']*0.45*4.57

df1 = df_org[['Species','VPD','PARin','SWC','Cond']]

df_combined = df1.append(df2)

df_combined.to_csv(r'D:/Academic/Alta Phd/E-ML models/df_combined.csv', index = False)

print(df_combined.info())

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

#x_standard = x_standard.fillna(x_standard.mean())
r,p = stats.pearsonr(y,x_standard['SWC'])

print('p: ',p)
print('r: ',r)

#create train-test split
x_train, x_test, y_train, y_test = (train_test_split(x_standard,y, test_size = 0.20, random_state = 20))

#create train-test split into different sizes
x_train20, x_test20, y_train20, y_test20 = (train_test_split(x_standard,y, test_size = 0.80, random_state = 20))
x_train30, x_test30, y_train30, y_test30 = (train_test_split(x_standard,y, test_size = 0.70, random_state = 20))
x_train40, x_test40, y_train40, y_test40 = (train_test_split(x_standard,y, test_size = 0.60, random_state = 20))
x_train50, x_test50, y_train50, y_test50 = (train_test_split(x_standard,y, test_size = 0.50, random_state = 20))
x_train60, x_test60, y_train60, y_test60 = (train_test_split(x_standard,y, test_size = 0.40, random_state = 20))
x_train70, x_test70, y_train70, y_test70 = (train_test_split(x_standard,y, test_size = 0.30, random_state = 20))
x_train80, x_test80, y_train80, y_test80 = (train_test_split(x_standard,y, test_size = 0.20, random_state = 20))
x_train90, x_test90, y_train90, y_test90 = (train_test_split(x_standard,y, test_size = 0.10, random_state = 20))


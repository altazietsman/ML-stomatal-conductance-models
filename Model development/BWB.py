from preprocessing import x_train,x_test,y_train,y_test, df1
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from numpy import exp
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import product


df1 = pd.read_excel('D:/Academic/Alta Phd/E-ML models/Input Data/df.xlsx')
df_anddereg = pd.read_csv('D:/Academic/Alta Phd/E-ML models/Input Data/AllData_EcologyLetters_Figshare_v1_318.csv')

#convert LWP to SWC
#the soil parameters for the van genuchten curves for loamy sand was used
df_anddereg = df_anddereg.replace(-9999.0, np.nan)

n = 1.56
m = 1-1/n
alpha = 0.036
df_anddereg['SWC_new'] = 1/((1+(-1*(df_anddereg['LWPpredawn'])/alpha)**n)**m)
df_anddereg.loc[df_anddereg['SWC'].isnull(),'SWC'] = df_anddereg['SWC_new']

df2 = df_anddereg[['Species','VPD','PARin','SWC','Cond','Photo','Tair','RH']]

df1['PARin'] = df1['solar']*0.45*4.57

#set parameters for An calculattion Farquhar model

ca = 40 #atmospheric CO2 partial presure in kpa
oa = 21000 #atmospheric O2 partial presure in kpa

ko25 = 28202 #michales-menton constant for oxylation (pa) at 25 degrees celcious
kc25 = 41 #michales-menton constant for carboxylation (pa) at 25 degrees celcious
qe = 0.3 #quatum yield of electron transport rate (mol photon mol-1 e)
cc = 0.9 #curvature of light response curve
cj = 0.98 #curvature of factor je v. jc limited photosynthesis (collatz et al 1991 and in CN)

#calculate An

#caluclate assimilation input parameters
df1['SVPD'] = ((610.78 * exp(df1['Tair'] / (df1['Tair']+238.3)*17.2694))/1000)/100 #maximum VPD in mol mol-1
df1['VPD'] = df1['SVPD']*(1-(df1['RH']/100)) #VPD in mol mol-1
df1['ea'] = (df1['SVPD'] - df1['VPD'])*df1['Patm'] #vapor presure deficit at atmospheric presure (kpa)
df1['eac'] = 1.72*((df1['ea']/(df1['Tair']+273.3))**(1/7)) #sky emmisivity (CN10:10)

vmax25 = 35
r = 4.36
jmax25 = 65


df1['vnumerator'] = vmax25*(1+exp((-4424)/(298*8.314)))*exp((73637/(8.314*298))*(1-(298/(273+df1['Tleaf']))))
df1['vdenominator'] = 1+exp((486*(df1['Tleaf']+273)-149252)/(8.314*(273+df1['Tleaf'])))
df1['vmax'] = df1['vnumerator']/df1['vdenominator'] #Vmax adjusted according to temperature (Leuning 2002 temperature adjusted)

df1['kc'] = kc25*exp((79430*((df1['Tleaf']+273)-298))/(298*8.314*(df1['Tleaf']+273))) #Kc temperature corrected (Bernacchi et al. 2001 temperature adjusted)
df1['ko'] = ko25*exp((36380*((df1['Tleaf']+273)-298))/(298*8.314*(df1['Tleaf']+273))) #Ko temperature corrected (Bernacchi et al. 2001 temperature adjusted)
df1['r'] = r*exp((37830*((df1['Tleaf']+273)-298))/(298*8.314*(df1['Tleaf']+273))) #temperature adjusted (Bernacchi et al. 2001)

df1['par'] = df1['solar']*0.45*4.57 #PAR (calculated from Wm-2) (umols-1m-2) Plant Growth Chamber Handbook (chapter 1, radiation; https://www.controlledenvironments.org/wp-content/uploads/sites/6/2017/06/Ch01.pdf)

df1['jnumerator'] = jmax25*(1+exp(-4534/(298*8.314)))*exp((50300/(8.314*298))*(1-(298/(df1['Tleaf']+273))))
df1['jdenominator'] = 1+exp((495*(df1['Tleaf']+273)-152044)/(8.314*(273+df1['Tleaf'])))
df1['jmax'] = df1['jnumerator']/df1['jdenominator'] #Jmax adjusted according to temperature (Leuning 2002 temperature adjusted)

df1['resp'] = (0.01*vmax25*2**(((df1['Tleaf']+273)-298)/10))* (1 + exp(1.3 * (df1['Tleaf'] - 55))) #respiration rate (umol s-1m-2 (collatz et a; 1991/ also in CN) and medlyn 2002

df1['ci'] = (ca/df1['Cond'])*df1['Patm'] #see moss & rowlins 1963

df1['Ac'] =(df1['vmax']*(df1['ci']-r))/(df1['ci']+df1['kc']*(oa/df1['ko']))-df1['resp']
df1['j'] =(qe*df1['par']+df1['jmax']-((((qe*df1['par']+df1['jmax'])**2)-(df1['par']*df1['jmax']*cc*qe*4))**0.5))/(2*cc)
df1['Aj'] = (df1['j']/4)*((df1['ci']-df1['r'])/(df1['ci']+2*df1['r']))

df1['Photo'] = df1[['Ac','Aj']].min(axis=1)


df_new = df1[['Species','VPD','PARin','SWC','Cond','Photo','Tair','RH']]

df_combined = df_new.append(df2)

#calculate RH from VPD where necessary
df_combined['SVPD'] = ((610.78 * exp(df_combined['Tair'] / (df_combined['Tair']+238.3)*17.2694))/1000)/100 #maximum VPD in mol mol-1
df_combined['RH_calculated'] = 100-(df_combined['VPD']/df_combined['SVPD'])
df_combined.RH.fillna(df_combined.RH_calculated, inplace=True)


#calculate ball_berry gw
g0 = 0.50

g1 = 15

ca = 40

df_combined['gsw_bb'] = (g0 + g1*((df_combined['Photo']*df_combined['RH'])/ca))/1000

df_combined.to_csv(r'D:/Academic/Alta Phd/E-ML models/df_bwb.csv', index = False)


#remove na

df_final = df_combined[df_combined['gsw_bb'].notna()]

#split into x and y

x = df_final[['PARin','RH','VPD','SWC','Tair','Photo']]
y = df_final['Cond']

#split into testing and training datasets

x_train, x_test, y_train, y_test = (train_test_split(x,y, test_size = 0.20, random_state = 20))

#grid search to find optimized fitted parameters (use only x data)

g0 = [0,0.5,1,1.5,2]
g1 = [5,10,15,20,25]

df_par = pd.DataFrame(list(product(g0, g1)), columns=['g0', 'g1'])
MSE_par = []
for i in range(0,5):
    gsw_bb = (df_par['g0'][i] + df_par['g1'][i] * ((x_train['Photo'] * x_train['RH']) / ca)) / 1000
    MSE = mean_squared_error(gsw_bb, y_train)
    MSE_par.append(MSE)

print('MSE min:', min(MSE_par))
index = MSE_par.index(min(MSE_par))
print('g0:', df_par['g0'][index])
print('g1:', df_par['g1'][index])

#from the grid search, the optimized parameters are g0 = 0 and g1 = 15
#get gsw_bb for test set based on optimized parameters

y_pred = (0 + 15 * ((x_test['Photo'] * x_test['RH']) / ca)) / 1000

plt.scatter(y_test, y_pred)
plt.show()

#calculate MSE
MSE = mean_squared_error(y_pred,y_test)

print("BWB MSW:", MSE)

print("BWB RMSE is:",np.sqrt(MSE))

#calculate R-squared
r2 = r2_score(y_test, y_pred)

print("BWB  r2 is:",r2)

#save pred. v. actual results

data_tuples = list(zip(y_test,y_pred))
df_BWB = pd.DataFrame(data_tuples, columns=['actual_gsw','bwb_gsw'])

df_BWB.to_csv(r'D:/Academic/Alta Phd/E-ML models/df_bwb_pred.csv', index = False)
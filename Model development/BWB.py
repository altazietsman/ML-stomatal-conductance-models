from preprocessing import x_train,x_test,y_train,y_test, df1
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from numpy import exp
from sklearn.model_selection import train_test_split
import pandas as pd


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
df1['SVPD'] = ((610.78 * exp(df1['temp'] / (df1['temp']+238.3)*17.2694))/1000)/100 #maximum VPD in mol mol-1
df1['VPD'] = df1['SVPD']*(1-(df1['RH']/100)) #VPD in mol mol-1
df1['ea'] = (df1['SVPD'] - df1['VPD'])*df1['P_atm'] #vapor presure deficit at atmospheric presure (kpa)
df1['eac'] = 1.72*((df1['ea']/(df1['temp']+273.3))**(1/7)) #sky emmisivity (CN10:10)

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

df1['ci'] = (ca/df1['gsw'])*df1['P_atm'] #see moss & rowlins 1963

df1['Ac'] =(df1['vmax']*(df1['ci']-r))/(df1['ci']+df1['kc']*(oa/df1['ko']))-df1['resp']
df1['j'] =(qe*df1['par']+df1['jmax']-((((qe*df1['par']+df1['jmax'])**2)-(df1['par']*df1['jmax']*cc*qe*4))**0.5))/(2*cc)
df1['Aj'] = (df1['j']/4)*((df1['ci']-df1['r'])/(df1['ci']+2*df1['r']))

df1['An'] = df1[['Ac','Aj']].min(axis=1)

#calculate ball_berry gw
g0 = 0.50

g1 = 15

ca = 40

df1['gsw_bb'] = (g0 + g1*((df1['An']*df1['RH'])/ca))/1000

#split it into same test set as for other models

x1, x_actual, y1, y_predict = (train_test_split(df1,df1['gsw_bb'], test_size = 0.1, random_state = 50))

#plot the predicted v. actual

x1 =pd.DataFrame(x_actual)

y_pred = list(y_predict)
y_act = list(x1['gsw'])


plt.plot(y_act)
plt.plot(y_pred)
plt.show()

#calculate MSE
MSE = mean_squared_error(y_pred,y_act)

print("BWB RMSE is:",np.sqrt(MSE))

#calculate R-squared
r2 = r2_score(y_act, y_pred)

print("BWB  r2 is:",r2)

#ARIMA Code:
pip install pmdarima

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Belfast2022-O3.csv')
df.info()
​
df.plot()
​
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 269 entries, 0 to 268
Data columns (total 1 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   O3      269 non-null    int64
dtypes: int64(1)
memory usage: 2.2 KB
<AxesSubplot:>

df
O3
0	60
1	59
2	61
3	56
4	35
...	...
264	58
265	64
266	40
267	57
268	26
269 rows × 1 columns

plt.figure(figsize=(20,10))
plt.xlabel('Days')
plt.ylabel('(V µg/m³')
plt.title('Levels of O3 in Belfast City Centre 2022')
plt.plot(df.O3, 'b.-', label='Ozone')
[<matplotlib.lines.Line2D at 0x28393547670>]

import numpy as np
msk = (df.index < len(df)-30)
df_train = df[msk].copy()
df_test = df[~msk].copy()
​
Check for stationarity of time series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
​
acf_original = plot_acf(df_train)
​
pacf_original = plot_pacf(df_train)
C:\Users\padra\anaconda3\lib\site-packages\statsmodels\graphics\tsaplots.py:348: FutureWarning: The default method 'yw' can produce PACF values outside of the [-1,1] interval. After 0.13, the default will change tounadjusted Yule-Walker ('ywm'). You can use this method now by setting method='ywm'.
  warnings.warn(


ADF test
from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(df_train)
print(f'p-value: {adf_test[1]}')
​
p-value: 2.722315088114772e-06
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
stepwise_fit = auto_arima(df['O3'], trace=True,
                          supress_warnings=True)
stepwise_fit.summary()
Performing stepwise search to minimize aic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=2096.784, Time=0.47 sec
 ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=2159.607, Time=0.02 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=2138.256, Time=0.06 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=2109.892, Time=0.06 sec
 ARIMA(0,1,0)(0,0,0)[0]             : AIC=2157.631, Time=0.02 sec
 ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.46 sec
 ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.43 sec
 ARIMA(3,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.70 sec
 ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.83 sec
 ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=2093.171, Time=0.17 sec
 ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=2095.110, Time=0.11 sec
 ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=2116.870, Time=0.10 sec
 ARIMA(1,1,1)(0,0,0)[0]             : AIC=2091.425, Time=0.08 sec
 ARIMA(0,1,1)(0,0,0)[0]             : AIC=2107.935, Time=0.04 sec
 ARIMA(1,1,0)(0,0,0)[0]             : AIC=2136.283, Time=0.05 sec
 ARIMA(2,1,1)(0,0,0)[0]             : AIC=2093.366, Time=0.16 sec
 ARIMA(1,1,2)(0,0,0)[0]             : AIC=2093.080, Time=0.18 sec
 ARIMA(0,1,2)(0,0,0)[0]             : AIC=2093.178, Time=0.06 sec
 ARIMA(2,1,0)(0,0,0)[0]             : AIC=2114.911, Time=0.06 sec
 ARIMA(2,1,2)(0,0,0)[0]             : AIC=2094.995, Time=0.27 sec

Best model:  ARIMA(1,1,1)(0,0,0)[0]          
Total fit time: 4.354 seconds
SARIMAX Results
Dep. Variable:	y	No. Observations:	269
Model:	SARIMAX(1, 1, 1)	Log Likelihood	-1042.712
Date:	Sun, 16 Apr 2023	AIC	2091.425
Time:	18:00:37	BIC	2102.198
Sample:	0	HQIC	2095.752
- 269		
Covariance Type:	opg		
coef	std err	z	P>|z|	[0.025	0.975]
ar.L1	0.4286	0.054	7.948	0.000	0.323	0.534
ma.L1	-0.9277	0.027	-35.009	0.000	-0.980	-0.876
sigma2	139.6626	11.411	12.239	0.000	117.297	162.028
Ljung-Box (L1) (Q):	0.01	Jarque-Bera (JB):	0.77
Prob(Q):	0.91	Prob(JB):	0.68
Heteroskedasticity (H):	0.75	Skew:	-0.02
Prob(H) (two-sided):	0.17	Kurtosis:	3.26


Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df_train, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                     O3   No. Observations:                  239
Model:                 ARIMA(1, 1, 1)   Log Likelihood                -921.324
Date:                Sun, 16 Apr 2023   AIC                           1848.648
Time:                        18:00:38   BIC                           1859.065
Sample:                             0   HQIC                          1852.847
                                - 239                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.3931      0.055      7.107      0.000       0.285       0.502
ma.L1         -0.9272      0.028    -33.102      0.000      -0.982      -0.872
sigma2       134.1834     11.150     12.034      0.000     112.329     156.038
===================================================================================
Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):                 2.16
Prob(Q):                              0.95   Prob(JB):                         0.34
Heteroskedasticity (H):               0.54   Skew:                            -0.01
Prob(H) (two-sided):                  0.01   Kurtosis:                         3.47
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
import matplotlib.pyplot as plt
residuals = model_fit.resid[1:]
fig, ax = plt.subplots(1,2)
residuals.plot(title='Residuals', ax=ax[0])
residuals.plot(title='Density', kind='kde', ax=ax[1])
plt.show()

acf_res = plot_acf(residuals)
​
pacf_res = plot_pacf(residuals)


forecast_test = model_fit.forecast(len(df_test))
​
df['forecast_manual'] = [None]*len(df_train) + list(forecast_test)
df.plot(figsize=(12,5),legend=True)
<AxesSubplot:>

import pmdarima as pm
auto_arima = pm.auto_arima(df_train, stepwise=False, seasonal=False)
auto_arima
ARIMA(order=(1, 1, 1), scoring_args={}, suppress_warnings=True)
auto_arima.summary()
SARIMAX Results
Dep. Variable:	y	No. Observations:	239
Model:	SARIMAX(1, 1, 1)	Log Likelihood	-921.242
Date:	Sun, 16 Apr 2023	AIC	1850.483
Time:	18:00:50	BIC	1864.373
Sample:	0	HQIC	1856.081
- 239		
Covariance Type:	opg		
coef	std err	z	P>|z|	[0.025	0.975]
intercept	-0.0228	0.059	-0.389	0.697	-0.138	0.092
ar.L1	0.3959	0.055	7.200	0.000	0.288	0.504
ma.L1	-0.9297	0.029	-32.563	0.000	-0.986	-0.874
sigma2	134.0752	11.189	11.982	0.000	112.144	156.006
Ljung-Box (L1) (Q):	0.00	Jarque-Bera (JB):	2.12
Prob(Q):	0.95	Prob(JB):	0.35
Heteroskedasticity (H):	0.54	Skew:	-0.02
Prob(H) (two-sided):	0.01	Kurtosis:	3.46


Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
mae = np.log(mean_absolute_error(df_test, forecast_test))
mape = np.log(mean_absolute_percentage_error(df_test, forecast_test))
rmse = np.log(np.sqrt(mean_squared_error(df_test, forecast_test)))
​
​
print(f'mae - manual: {mae}')
print(f'mape - manual: {mape}')
print(f'rmse - manual: {rmse}')
mae - manual: 2.6240109279299606
mape - manual: 0.24698880109343443
rmse - manual: 2.8246743170111825

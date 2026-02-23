# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 23/20/2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
data=pd.read_csv('AirPassengers.csv')
N=1000
plt.rcParams['figure.figsize'] = [12, 6] #plt.rcParams is a dictionary-like object in Mat
X=data['#Passengers']
plt.plot(X)
plt.title('Original Data')
plt.show()
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])  
ma2 = np.array([1, theta1_arma22, theta2_arma22])  
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
```
OUTPUT:
<img width="971" height="540" alt="image" src="https://github.com/user-attachments/assets/e9762845-4f2c-4e9d-8ee1-f42392376171" />

<img width="985" height="509" alt="image" src="https://github.com/user-attachments/assets/7fa8ef57-4f39-4ac1-a12d-b6584cb340c3" />

SIMULATED ARMA(1,1) PROCESS:

<img width="977" height="526" alt="image" src="https://github.com/user-attachments/assets/efbe7bb0-37e7-4e5b-a15f-b0cba801c6e5" />


Partial Autocorrelation

<img width="981" height="521" alt="image" src="https://github.com/user-attachments/assets/3a1fb323-6631-406a-83ec-dc381441087e" />


Autocorrelation
<img width="976" height="528" alt="image" src="https://github.com/user-attachments/assets/3147e557-7a0d-40b2-97a2-412a00947409" />



SIMULATED ARMA(2,2) PROCESS:
<img width="965" height="537" alt="image" src="https://github.com/user-attachments/assets/0de283e4-1762-4889-a30a-0ba4533dcb3e" />


Partial Autocorrelation
<img width="971" height="533" alt="image" src="https://github.com/user-attachments/assets/b3bddf3d-c345-4115-b0ed-a8c4513763ad" />



Autocorrelation

<img width="993" height="517" alt="image" src="https://github.com/user-attachments/assets/53fa10fb-0974-4c53-8fe9-63d08aa53c3a" />


RESULT:
Thus, a python program is created to fir ARMA Model successfully.

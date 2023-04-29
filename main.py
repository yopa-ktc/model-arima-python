import pandas as pd
import numpy as np
from model_prepare import ModelPrepare

df = pd.read_csv('website_data.csv')


newModelPrepare = ModelPrepare(df)

newModelPrepare.visualisation()
#Utiliser le logarithme sur les éléments du data pour normaliser les données
df_log = np.log(df)
#df_log.plot()
#plt.show()


msk = (df.index < len(df)-30)
df_train = df_log[msk].copy()
df_test = df_log[~msk].copy()


#Checker la stationnarité
#Nous traçons la courbe et nous l'observons
#Si c'est plus difficile, nous traçons ACF ou PACF

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#acf_original = plot_acf(df_train)
#pacf_original = plot_pacf(df_train)


#VERIFICATION AVEC ADF
from statsmodels.tsa.stattools import adfuller
#adf_test = adfuller(df_train)
#print(f'p-value: {adf_test[1]}')
#Le test donne une valeur différente de 0 ce qui signifie
#que l'hypothèse de la non stationnarité est confirmée.

#----------

#Transformons notre série chronologique en série stationnaire
#Utilisons la fonction dropna()
df_train_diff = df_train.diff().dropna()
#df_train_diff.plot()
#Retraçons les ACF et PACF
#acf_diff = plot_acf(df_train_diff)
#pacf_diff = plot_pacf(df_train_diff)
#Recalculons l'ADF 
adf_test_diff = adfuller(df_train_diff)
#print(f'p-value: {adf_test_diff[1]}')
#plt.show()

"""
#___La p-value se rapproche encore plus de 0 ce qui
#signifie bien que notre modèle peut bien être stationnaire
#On peut laisser penser qu'à la 2e itération de différence,
#La stationnarité pourra encore s'améliorer
#_______________

#On peut à présent déterminer le modèle ARIMA
#_____Les paramètres P et Q
#_____Le modèle (2,1,0)


from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df_train, order=(2,1,0))
#Entrainons notre modèle
model_fit = model.fit()
#print(model_fit.summary())

#Vérifions la qualité de notre modèle à l'aide des résidus
#Si le modèle est bon, les résidus devraient ressembler à du bruit.

residus = model_fit.resid[1:]

#fig, ax = plt.subplots(1,2)
#residus.plot(title='Residuals', ax=ax[0])
#residus.plot(title='Density', kind='kde', ax=ax[1])


#On trace encore l'ACF et l'PACF des résidus
#acf_resid = plot_acf(residus)
#pacf_resid = plot_pacf(residus)
#Le tracé montre bien que les résidus sont proches du bruit blanc
#plt.show()

#Nous sommes donc prêts à faire nos prédictions
#Autrement dit, faisons le forecast

forecast_test = model_fit.forecast(len(df_test))

df_log['forecast_manual'] = [None]*len(df_train) + list(forecast_test)


import pmdarima as pm
auto_arima = pm.auto_arima(df_train, stepwise=False, seasonal=True)

forecast_test_auto = auto_arima.predict(n_periods=len(df_test))

df_log['forecast_auto'] = [None]*len(df_train) + list(forecast_test_auto)
df_log.plot()
plt.show()
"""
import pandas as pd
from model_prepare import ModelPrepare
from arima_model import ArimaModel

df = pd.read_csv('website_data.csv')

newModelPrepare = ModelPrepare(df)


#newModelPrepare.visualisation()


df_log = newModelPrepare.normalisation()


df_train, df_test = newModelPrepare.decoupage(df_log)

"""
    Checker la stationnarité
    Nous traçons la courbe et nous l'observons
    Si c'est plus difficile, nous traçons ACF ou PACF
"""
#acf_original, pacf_original = newModelPrepare.showACF(df_train)


"""
    VERIFICATION AVEC ADF
    Le test donne une valeur différente de 0 ce qui signifie
    que l'hypothèse de la non stationnarité est confirmée.
"""
newModelPrepare.showADF(df_train)


"""
    Transformons notre série chronologique en série stationnaire
    Utilisons la fonction dropna()
"""
#df_train_diff = df_train.diff().dropna()
#df_train_diff.plot()

#___Retraçons les ACF et PACF
#acf_diff, pacf_diff = newModelPrepare.showACF(df_train_diff)
#___Recalculons l'ADF 
#adf_test_diff = newModelPrepare.showADF(df_train)


"""
    La p-value se rapproche encore plus de 0 ce qui
    signifie bien que notre modèle peut bien être stationnaire
    On peut laisser penser qu'à la 2e itération de différence,
    La stationnarité pourra encore s'améliorer

    On peut à présent déterminer le modèle ARIMA
    Les paramètres P et Q
    Le modèle (2,1,0)
"""
newArima = ArimaModel(df_log, df_train, df_test)
model_fit = newArima.fitModel()
#print(model_fit.summary())


"""
    #Vérifions la qualité de notre modèle à l'aide des résidus
    #Si le modèle est bon, les résidus devraient ressembler à du bruit.
"""
#residus = newArima.getResidus(model_fit)


"""
    On trace encore l'ACF et l'PACF des résidus
    Le tracé montre bien que les résidus sont proches du bruit blanc
"""
#acf_resid, pacf_resid = newModelPrepare.showACF(residus)

"""
    Nous sommes donc prêts à faire nos prédictions
    Autrement dit, faisons le forecast
"""
newArima.forecastTest(model_fit)


"""
import pmdarima as pm
auto_arima = pm.auto_arima(df_train, stepwise=False, seasonal=True)
forecast_test_auto = auto_arima.predict(n_periods=len(df_test))
df_log['forecast_auto'] = [None]*len(df_train) + list(forecast_test_auto)
df_log.plot()
plt.show()
"""
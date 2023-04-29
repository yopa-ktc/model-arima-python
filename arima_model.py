from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pmdarima as pm

class ArimaModel:
    
    def __init__(self, df_log, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test
        self.df_log = df_log
        self.model_fit = self.fitModel()
        
    def fitModel(self):
        model = ARIMA(self.df_train, order=(2,1,0))
        model_fit = model.fit()        
        return model_fit
    
    def getResidus(self):
        residus = self.model_fit.resid[1:]
        fig, ax = plt.subplots(1,2)
        residus.plot(title='Residuals', ax=ax[0])
        residus.plot(title='Density', kind='kde', ax=ax[1])
        plt.show()
        return residus
    
    def forecastTest(self):
        forecast_test = self.model_fit.forecast(len(self.df_test))
        self.df_log['forecast_manual'] = [None]*len(self.df_train) + list(forecast_test)
        self.df_log.plot()
        plt.show()
        
    def forecastAuto(self):
        auto_arima = pm.auto_arima(self.df_train, stepwise=False, seasonal=True)
        forecast_test_auto = auto_arima.predict(n_periods=len(self.df_test))
        self.df_log['forecast_auto'] = [None]*len(self.df_train) + list(forecast_test_auto)
        self.df_log.plot()
        plt.show()
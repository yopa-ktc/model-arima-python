from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

class ArimaModel:
    
    def __init__(self, df_log, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test
        self.df_log = df_log
        self.fitModel()
        
    def fitModel(self):
        model = ARIMA(self.df_train, order=(2,1,0))
        model_fit = model.fit()        
        return model_fit
    
    def getResidus(self, model_fit):
        residus = model_fit.resid[1:]
        fig, ax = plt.subplots(1,2)
        residus.plot(title='Residuals', ax=ax[0])
        residus.plot(title='Density', kind='kde', ax=ax[1])
        plt.show()
        return residus
    
    def forecastTest(self, model_fit):
        forecast_test = model_fit.forecast(len(self.df_test))
        self.df_log['forecast_manual'] = [None]*len(self.df_train) + list(forecast_test)
        plt.show()
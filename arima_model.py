from statsmodels.tsa.arima.model import ARIMA

class ArimaModel:
    
    def __init__(self, df_train):
        self.df_train = df_train
        self.fitModel()
        
    def fitModel(self):
        model = ARIMA(self.df_train, order=(2,1,0))
        #Entrainons notre modèle
        model_fit = model.fit()
        print(model_fit.summary())
        return model_fit
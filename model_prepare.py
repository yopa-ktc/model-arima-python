import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

class ModelPrepare:
    
    def __init__(self,dataset):
        self.dataset = dataset
    
    def visualisation(self):
        #Visualiser la donnée
        self.dataset.plot()
        plt.show()
        
    def normalisation(self): #Utiliser le logarithme sur les éléments du data pour normaliser les données
        df_log = np.log(self.dataset)
        df_log.plot()
        plt.show()
        return df_log
    
    def decoupage(self, df_log):
        msk = (self.dataset.index < len(self.dataset)-30)
        df_train = df_log[msk].copy()
        df_test = df_log[~msk].copy()
        return df_train, df_test
    
    def showACF(self, df_train):
        acf = plot_acf(df_train)
        pacf = plot_pacf(df_train)
        plt.show()
        return acf, pacf
    
    def showADF(self, df_train):
        adf_test = adfuller(df_train)
        print(f'p-value: {adf_test[1]}')
        return adf_test[1]
    
    



class ModelPrepare:
    
    def __init__(self,dataset):
        self.dataset = dataset
    
    def visualition(self):
        #Visualiser la donnée
        self.dataset.plot()
        plt.show()

#__________ACF : Correlation totale
#__________PACF : Corelation partielle
#__________ADF : 
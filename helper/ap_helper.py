import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class Helper():
    def CriarPasta(pDiretorioDestino: str):
        os.makedirs(pDiretorioDestino, exist_ok = True)  
        lDestino = pDiretorioDestino 
        return lDestino
    
    def MostrarPerdaPlot(pHistorico, pNomeModelo: str):
        plt.plot(pHistorico.history['loss'], label = 'Treino')
        plt.plot(pHistorico.history['val_loss'], label = 'Validação')
        plt.title(f'{pNomeModelo} - Perda')
        plt.xlabel('Épocas')
        plt.ylabel('Perda')
        plt.legend()
        plt.show()
    
    def MostrarMatrixConfusao(pMatriz, pClasses, pNomeModelo):
        lMatiz = pd.DataFrame(pMatriz, index = pClasses[:len(pMatriz)], columns = pClasses[:len(pMatriz)])
        
        plt.figure(figsize = (10, 8))
        sns.heatmap(lMatiz, annot = True, cmap = 'Blues')  
        plt.title(f'{pNomeModelo} - Matriz de Confusão')
        plt.xlabel('Previsão')
        plt.ylabel('Real')
        plt.show()

    def DiretorioAtual():
        lDiretorio = os.path.dirname(os.path.realpath(__file__)) 
        return os.path.dirname(lDiretorio)      
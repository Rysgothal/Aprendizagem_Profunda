import os
import matplotlib.pyplot as plt

class Helper():
    def CriarPasta(pDiretorioDestino: str):
        """Função onde cria uma pasta

        Args:
            pDiretorioDestino (str): Caminho onde a pasta vai ser criada

        Returns:
            lDestino (str): Local onde a pasta foi criada
        """
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
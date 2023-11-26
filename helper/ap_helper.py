import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import tensorflow as tf

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

class Modelo():
    def __init__(self, pNome: str):
        self.Modelo = tf.keras.Sequential(name = pNome) 

    def AdicionarEmbedding(self, pInputDim: int, pInputLength: int):
        lOutputDim = 32             # fixo valor 32
        lModelo = self.Modelo
        lModelo.add(tf.keras.layers.Embedding(input_dim = pInputDim, output_dim = lOutputDim, input_length = pInputLength))
        
        self.Modelo = lModelo

    def AdicionarInput(self, pShape: int):
        self.Modelo.add(tf.keras.layers.Input(shape = pShape))

    def AdicionarPooling(self):
        self.Modelo.add(tf.keras.layers.GlobalAveragePooling1D())  # Camada de pooling

    def AdicionarDense(self, pQtdeNeuronios: int = 64, pAtivacao: str = 'relu'):
        self.Modelo.add(tf.keras.layers.Dense(pQtdeNeuronios, activation = pAtivacao))

    def AdicionarDropOut(self, pValor: float):
        self.Modelo.add(tf.keras.layers.Dropout(pValor))

    def AdicionarLSTM(self, pValor: int, pRetornar: bool):
        self.Modelo.add(tf.keras.layers.LSTM(pValor, return_sequences = pRetornar))

    # def AdicionarConvUnidirecional(self, pNumFiltros: int, pAtivacao: str):
    #     self.Modelo.add(tf.keras.layers.Conv1D(filters = pNumFiltros, activation = pAtivacao))

    # def AdicionarConvUnidirecional(self, pNumFiltros: int, pTamanhoConv: int, pAtivacao: str):
    #     self.Modelo.add(tf.keras.layers.Conv1D(filters = pNumFiltros, kernel_size = pTamanhoConv, activation = pAtivacao))

    def AdicionarConvUnidirecional(self, pNumFiltros: int, pAtivacao: str, pTamanhoConv: int = None, pPasso: int = None):
        if pPasso is not None and pTamanhoConv is not None:
            self.Modelo.add(tf.keras.layers.Conv1D(pNumFiltros, pTamanhoConv, strides = pPasso, activation = pAtivacao))
            
        elif pTamanhoConv is not None:
            self.Modelo.add(tf.keras.layers.Conv1D(pNumFiltros, pTamanhoConv, activation = pAtivacao))
            
        else:
            self.Modelo.add(tf.keras.layers.Conv1D(pNumFiltros, activation = pAtivacao))


    def Compilar(self, pPerda: str, pOtimizador, pMetrica: list[str]):
        self.Modelo.compile(loss = pPerda, optimizer = pOtimizador, metrics = pMetrica)

    def CustomizarOtimizador(self, pTaxaAprendizado: str):
        lOtimizador = tf.keras.optimizers.Adam(learning_rate = pTaxaAprendizado)
        return lOtimizador
    
    def Treinamento(self, pSequenciaEntrada, pLinguasCodificadas, pDadosValidacao: tuple, pEpocas = 500, pTamanhoLote = 1):
        lHistorico = self.Modelo.fit(pSequenciaEntrada, pLinguasCodificadas, 
                                     epochs = pEpocas, batch_size = pTamanhoLote,
                                     validation_data = pDadosValidacao)
        
        return lHistorico
    
    def Salvar(self, pNome: str):
        self.Modelo.save(Helper.DiretorioAtual() + '/Models/' + f'{pNome}.keras')

        # lModeloTamanhoFiltro.add(tf.keras.layers.Conv1D(128, lTamanhoFiltro, activation = 'relu'))  # Variando o tamanho do filtro conforme desejado

        # lModeloFiltroPasso.add(tf.keras.layers.Conv1D(128, 5, strides = lFiltroPasso, activation = 'relu'))  # Variando o tamanho do passo conforme desejado


        # lOtimizador = tf.keras.optimizers.Adam(learning_rate = 0.001)
        # lModeloDNN4.compile(loss = 'sparse_categorical_crossentropy', optimizer = lOtimizador, metrics = ['accuracy'])

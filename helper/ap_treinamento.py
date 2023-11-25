import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from helper.ap_helper import Helper 

def CarregarLinguagemDataSet(pCaminho: str):
    lFrases = []
    lLingua = []

    for lCategoria in os.listdir(pCaminho):
        lCaminhoCategoria = os.path.join(pCaminho, lCategoria)

        if not os.path.isdir(lCaminhoCategoria):
            continue

        for lNomeArquivo in os.listdir(lCaminhoCategoria):
            lCaminhoArquivo = os.path.join(lCaminhoCategoria, lNomeArquivo)

            if not os.path.isfile(lCaminhoArquivo):
                continue
        
            with open(lCaminhoArquivo, 'r', encoding = 'utf-8') as lArquivo:
                lFrases.append(lArquivo.read())
                lLingua.append(lCategoria)

    return lFrases, lLingua

# Carregando o Dataset de treinamento e teste
lTreinoPasta = Helper.DiretorioAtual() + '/helper/Textos/train' 
lTestePasta = Helper.DiretorioAtual() + '/helper/Textos/test' 
lTreinoFrases, lTreinoLinguas = CarregarLinguagemDataSet(lTreinoPasta)
lTesteFrases, lTesteLinguas = CarregarLinguagemDataSet(lTestePasta)

# Combinando as frases e linguas de treinamento e teste
lFrases = lTreinoFrases + lTesteFrases 
lLinguas = lTreinoLinguas + lTesteLinguas 

# Pré-processamento de texto
lTokenizador = tf.keras.layers.TextVectorization(max_tokens = 20000, output_mode = "int")
lTokenizador.adapt(lFrases)

# Pré-processamento dos dados de Testes e Treinos
lSequencias = lTokenizador(lFrases)
lSequenciasTreino = lTokenizador(np.array(lTreinoFrases)).numpy()
lSequenciasTeste = lTokenizador(np.array(lTesteFrases)).numpy()

lTamanhoMaximo = max(len(lSeq) for lSeq in lSequenciasTreino)
lSequenciasTreino = tf.keras.preprocessing.sequence.pad_sequences(lSequenciasTreino, maxlen = lTamanhoMaximo, padding = 'post')
lSequenciasTeste = tf.keras.preprocessing.sequence.pad_sequences(lSequenciasTeste, maxlen = lTamanhoMaximo, padding = 'post')

# Tamanho do vocabulário
lTamanhoVocabulario = len(lTokenizador.get_vocabulary())

# Codificação das etiquetas
lCodificarLinguas = LabelEncoder()
lCodificarLinguas.fit(lTreinoLinguas)
lLinguasTesteCodificado = lCodificarLinguas.transform(lTesteLinguas)
lLinguasTreinoCodificado = lCodificarLinguas.transform(lTreinoLinguas)
lSequenciasTreino = lTokenizador(lTreinoFrases)
lTamanhoMaximo = max(len(lSeq) for lSeq in lSequenciasTreino)
    
# Modelo ANNs (Redes Neurais Artificiais) com dropout

lModeloDropOut = tf.keras.Sequential()
lModeloDropOut.add(tf.keras.layers.Embedding(input_dim = lTamanhoVocabulario, output_dim = 32, input_length = len(lSequencias[0])))
lModeloDropOut.add(tf.keras.layers.GlobalAveragePooling1D())  # Camada de pooling
lModeloDropOut.add(tf.keras.layers.Dense(64, activation = 'relu'))
lModeloDropOut.add(tf.keras.layers.Dropout(0.5))
lModeloDropOut.add(tf.keras.layers.Dense(len(set(lLinguas)), activation = 'softmax'))
lModeloDropOut.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Treinando o modelo usando as sequências de treinamento e rótulos codificados
lHistoricoDropOut = lModeloDropOut.fit(lSequenciasTreino, lLinguasTreinoCodificado, epochs = 500, batch_size = 1,
    validation_data = (lSequenciasTeste, lLinguasTesteCodificado))

# Avaliação do modelo usando os dados de teste
lDropOutPredicaoY = lModeloDropOut.predict(lSequenciasTeste)
lDropOutPredicaoClassesY = np.argmax(lDropOutPredicaoY, axis = 1)
lDropOutAcuracia = accuracy_score(lLinguasTesteCodificado, lDropOutPredicaoClassesY)
print(f"Acurácia do modelo ANNs com dropout: {lDropOutAcuracia}")

# Plotando o gráfico de perda e a matriz de confusão
Helper.MostrarPerdaPlot(lHistoricoDropOut, 'ANN Dropout Model')

lClassesUnique = np.unique(lLinguas)
print("Classes: ", lClassesUnique)
lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lDropOutPredicaoClassesY, normalize = 'true')
print("Matriz:", lMatrizConfusao)
Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'ANN Dropout Model')

lModeloDropOut.save(Helper.DiretorioAtual() + '/Models/modelANNs_dropout.keras')


# Variações de hiperparâmetros para o modelo ANNs
lUnidadesANN = [32, 64, 128]

# Criando, treinando e avaliando modelo ANNs para cada variação de unidades ANNs
for lUnidadeANN in lUnidadesANN:
    # Criando um modelo ANNs com a variação do número de unidades ANNs
    lModeloANN = tf.keras.Sequential()
    lModeloANN.add(tf.keras.layers.Embedding(input_dim = lTamanhoVocabulario, output_dim = 32, input_length = len(lSequencias[0])))
    lModeloANN.add(tf.keras.layers.GlobalAveragePooling1D())  # Camada de pooling
    lModeloANN.add(tf.keras.layers.Dense(lUnidadeANN, activation = 'relu'))
    lModeloANN.add(tf.keras.layers.Dense(len(set(lLinguas)), activation = 'softmax'))
    lModeloANN.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    # Treinando o modelo
    lHistoricoANN = lModeloANN.fit(lSequenciasTreino, lLinguasTreinoCodificado, epochs = 500, batch_size = 1, validation_data = (lSequenciasTeste, lLinguasTesteCodificado))
    
    lPredicaoY = lModeloANN.predict(lSequenciasTeste)
    lPredicaoClassesY = np.argmax(lPredicaoY, axis = 1)
    lAcuracia = accuracy_score(lLinguasTesteCodificado, lPredicaoClassesY)

    print(f"Acurácia do modelo ANNs ({lUnidadeANN}): {lAcuracia}")
    Helper.MostrarPerdaPlot(lHistoricoANN, 'ANN Model')

    lClassesUnique = np.unique(lLinguas)
    lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoClassesY)
    Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'ANN Model')

    lModeloANN.save(Helper.DiretorioAtual() + f'/Models/modelANNs_{lUnidadeANN}.keras')


# Terceiro Modelo ANNs com Camada LSTM
lModeleCamadaLSTM = tf.keras.Sequential()
lModeleCamadaLSTM.add(tf.keras.layers.Embedding(input_dim = lTamanhoVocabulario, output_dim = 32, input_length = len(lSequencias[0])))
lModeleCamadaLSTM.add(tf.keras.layers.LSTM(64, return_sequences = True))  # Camada LSTM
lModeleCamadaLSTM.add(tf.keras.layers.GlobalMaxPooling1D())  # Camada de pooling
lModeleCamadaLSTM.add(tf.keras.layers.Dense(len(set(lLinguas)), activation = 'softmax'))
lModeleCamadaLSTM.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Treinando o modelo LSTM
lHistoricoANNCamadaLSTM = lModeleCamadaLSTM.fit(lSequenciasTreino, lLinguasTreinoCodificado, epochs = 500, batch_size = 1, validation_data = (lSequenciasTeste, lLinguasTesteCodificado))

# Avaliando do modelo LSTM
lPredicaoYCamadaLSTM = lModeleCamadaLSTM.predict(lSequenciasTeste)
lPredicaoYCamadaLSTMClasses = np.argmax(lPredicaoYCamadaLSTM, axis = 1)

lAcuraciaLSTM = accuracy_score(lLinguasTesteCodificado, lPredicaoYCamadaLSTMClasses)
print(f"Acurácia do modelo ANNs com LSTM: {lAcuraciaLSTM}")
Helper.MostrarPerdaPlot(lHistoricoANNCamadaLSTM, 'ANN LSTM Model')

lClassesUnique = np.unique(lLinguas)
lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoYCamadaLSTMClasses)
Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'ANN LSTM Model')

lModeleCamadaLSTM.save(Helper.DiretorioAtual() + '/Models/modelANNs_lstm.keras')

# Variações de hiperparâmetros para o modelo CNN
lNumeroFiltrosLista = [64, 128, 256]

# Criando, treinando e avaliando um modelo CNN para cada variação de número de filtros
for lNumerosFitros in lNumeroFiltrosLista:
    # Criando um modelo CNN com a variação do número de filtros
    lModeloCNN = tf.keras.Sequential()
    lModeloCNN.add(tf.keras.layers.Embedding(input_dim=lTamanhoVocabulario, output_dim=32, input_length=len(lSequencias[0])))
    lModeloCNN.add(tf.keras.layers.Conv1D(lNumerosFitros, 5, activation = 'relu'))
    lModeloCNN.add(tf.keras.layers.GlobalMaxPooling1D())
    lModeloCNN.add(tf.keras.layers.Dense(len(set(lLinguas)), activation = 'softmax'))
    lModeloCNN.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # Treinando o modelo
    lHistoricoCNNFiltro = lModeloCNN.fit(lSequenciasTreino, lLinguasTreinoCodificado, epochs = 500, batch_size = 1, validation_data = (lSequenciasTeste, lLinguasTesteCodificado))

    # Avaliando do modelo
    lPredicaoY = lModeloCNN.predict(lSequenciasTeste)
    lPredicaoClassesY = np.argmax(lPredicaoY, axis=1)
    
    lAcuracia = accuracy_score(lLinguasTesteCodificado, lPredicaoClassesY)
    print(f"Acurácia do modelo CNN ({lNumerosFitros} filtros): {lAcuracia}")
    Helper.MostrarPerdaPlot(lHistoricoCNNFiltro ,  'CNN Filter Model')
    
    lClassesUnique = np.unique(lLinguas)
    lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoClassesY)
    Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'CNN Filter Model')
    
    lModeloCNN.save(Helper.DiretorioAtual() + f'/Models/modelCNN_{lNumerosFitros}.keras')

# Variação de outro hiperparâmetro para os modelos CNN
# Modelo 1 com tamanho de filtro diferente
lTamanhoFiltroLista = [3, 5, 7]

for lTamanhoFiltro in lTamanhoFiltroLista:
    # Criando um modelo CNN com a variação do tamanho do filtro
    lModeloTamanhoFiltro = tf.keras.Sequential()
    lModeloTamanhoFiltro.add(tf.keras.layers.Embedding(input_dim = lTamanhoVocabulario, output_dim = 32, input_length = len(lSequencias[0])))
    lModeloTamanhoFiltro.add(tf.keras.layers.Conv1D(128, lTamanhoFiltro, activation = 'relu'))  # Variando o tamanho do filtro conforme desejado
    lModeloTamanhoFiltro.add(tf.keras.layers.GlobalMaxPooling1D())
    lModeloTamanhoFiltro.add(tf.keras.layers.Dense(len(set(lLinguas)), activation = 'softmax'))
    lModeloTamanhoFiltro.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # Treinando o modelo
    lHistoricoFiltroTamanhoCNN = lModeloTamanhoFiltro.fit(lSequenciasTreino, lLinguasTreinoCodificado, epochs = 500, batch_size = 1, validation_data = (lSequenciasTeste, lLinguasTesteCodificado))

    # Salvando o modelo com um nome indicando a variação do hiperparâmetro
    lModeloTamanhoFiltro.save(Helper.DiretorioAtual() + f'/Models/modelCNN_filter_size_{lTamanhoFiltro}.keras')

    # Avaliação do modelo
    lPredicaoY = lModeloTamanhoFiltro.predict(lSequenciasTeste)
    lPredicaoClassesY = np.argmax(lPredicaoY, axis=1)
    lAcuracia = accuracy_score(lLinguasTesteCodificado, lPredicaoClassesY)
    print(f"Acurácia do modelo CNN (filtro de tamanho {lTamanhoFiltro}): {lAcuracia}")
    Helper.MostrarPerdaPlot(lHistoricoFiltroTamanhoCNN, 'CNN Filter Size Model')
    lClassesUnique = np.unique(lLinguas)
    lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoClassesY)
    Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'CNN Filter Size Model')
    

# Modelo 2 com tamanho do passo diferente
lFiltroPassosLista = [1, 2, 3]

for lFiltroPasso in lFiltroPassosLista:
    # Criando um modelo CNN com a variação do tamanho do passo
    lModeloFiltroPasso = tf.keras.Sequential()
    lModeloFiltroPasso.add(tf.keras.layers.Embedding(input_dim = lTamanhoVocabulario, output_dim = 32, input_length = len(lSequencias[0])))
    lModeloFiltroPasso.add(tf.keras.layers.Conv1D(128, 5, strides = lFiltroPasso, activation = 'relu'))  # Variando o tamanho do passo conforme desejado
    lModeloFiltroPasso.add(tf.keras.layers.GlobalMaxPooling1D())
    lModeloFiltroPasso.add(tf.keras.layers.Dense(len(set(lLinguas)), activation = 'softmax'))
    lModeloFiltroPasso.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # Treinando o modelo
    lHistoricoFiltroCNNPasso = lModeloFiltroPasso.fit(lSequenciasTreino, lLinguasTreinoCodificado, epochs = 500, batch_size = 1, validation_data = (lSequenciasTeste, lLinguasTesteCodificado))

    # Salvando o modelo com um nome indicando a variação do hiperparâmetro
    lModeloFiltroPasso.save(Helper.DiretorioAtual() + f'/Models/modelCNN_filter_stride_{lFiltroPasso}.keras')

    # Avaliação do modelo
    lPredicaoY = lModeloFiltroPasso.predict(lSequenciasTeste)
    lPredicaoClassesY = np.argmax(lPredicaoY, axis = 1)
    lAcuracia = accuracy_score(lLinguasTesteCodificado, lPredicaoClassesY)
    print(f"Acurácia do modelo CNN (passo de tamanho {lFiltroPasso}): {lAcuracia}")
    Helper.MostrarPerdaPlot(lHistoricoFiltroCNNPasso , 'CNN Filter Stride Model')
    lClassesUnique = np.unique(lLinguas)
    lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoClassesY)
    Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'CNN Filter Stride Model')

# Modelo DNNs (Rede Neural Profunda) com número de camadas ocultas
LModeloLayers = tf.keras.Sequential(name = "model_layers")
LModeloLayers.add(tf.keras.layers.Input(shape = (len(lSequencias[0]), )))  # Alterando o tamanho da sequência de acordo
LModeloLayers.add(tf.keras.layers.Dense(64, activation = 'relu'))
LModeloLayers.add(tf.keras.layers.Dense(64, activation = 'relu'))
LModeloLayers.add(tf.keras.layers.Dense(64, activation = 'relu'))
LModeloLayers.add(tf.keras.layers.Dense(len(set(lLinguas)), activation = 'softmax'))
LModeloLayers.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Treinando e avaliando o modelo Model_Layers
lHistoricoDNNOCamadaOculta = LModeloLayers.fit(lSequenciasTreino, lLinguasTreinoCodificado, epochs = 500, batch_size = 1, validation_data = (lSequenciasTeste, lLinguasTesteCodificado))
lPredicaoLayersY = LModeloLayers.predict(lSequenciasTeste)
lPredicaoLayersYClasses = np.argmax(lPredicaoLayersY, axis = 1)
lAcuraciaLayers = accuracy_score(lLinguasTesteCodificado, lPredicaoLayersYClasses)
print(f"Acurácia do modelo LModeloLayers: {lAcuraciaLayers}")
Helper.MostrarPerdaPlot(lHistoricoDNNOCamadaOculta, 'DNN Ocult Model')
lClassesUnique = np.unique(lLinguas)
lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoLayersYClasses)
Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'DNN Ocult Model')
# Salvando o modelo LModeloLayers no diretório atual
LModeloLayers.save(Helper.DiretorioAtual() + '/Models/modelDNN_layers.keras')


# Modelo DNNs (Rede Neural Profunda) com diferentes funções de ativação nas camadas ocultas
lModeloDNN = tf.keras.Sequential(name = "modelDNN")
lModeloDNN.add(tf.keras.layers.Input(shape=(len(lSequencias[0]),)))  # Alterando o tamanho da sequência de acordo
lModeloDNN.add(tf.keras.layers.Dense(64, activation = 'relu'))
lModeloDNN.add(tf.keras.layers.Dense(64, activation = 'tanh'))  # Variando a função de ativação conforme desejado
lModeloDNN.add(tf.keras.layers.Dense(64, activation = 'sigmoid'))
lModeloDNN.add(tf.keras.layers.Dense(len(set(lLinguas)), activation = 'softmax'))
lModeloDNN.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Treinando e avaliando o modelo lModeloDNN
lHistoricoDNNCamadaOculta = lModeloDNN.fit(lSequenciasTreino, lLinguasTreinoCodificado, epochs = 500, batch_size = 1, validation_data = (lSequenciasTeste, lLinguasTesteCodificado))

lDNNPredicaoY = lModeloDNN.predict(lSequenciasTeste)
lPredicaoDNNClassesY = np.argmax(lDNNPredicaoY, axis = 1)
lAcuraciaDNN = accuracy_score(lLinguasTesteCodificado, lPredicaoDNNClassesY)
print(f"Acurácia do modelo lModeloDNN: {lAcuraciaDNN}")
Helper.MostrarPerdaPlot(lHistoricoDNNCamadaOculta, 'DNN Ocult Ativation Model')
lClassesUnique = np.unique(lLinguas)
lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoDNNClassesY)
Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'DNN Ocult Ativation Model')

# Salvando o modelo lModeloDNN
lModeloDNN.save(Helper.DiretorioAtual() + '/Models/lModeloDNN.keras')


# Modelo DNNs (Rede Neural Profunda) com variação da taxa de aprendizado e função de ativação
lModeloDNN4 = tf.keras.Sequential(name = "modelDNN4")
lModeloDNN4.add(tf.keras.layers.Input(shape = (len(lSequencias[0]),)))  # Ajuste o tamanho da sequência de acordo
lModeloDNN4.add(tf.keras.layers.Dense(64, activation = 'relu'))  # Mantenha o número de unidades
lModeloDNN4.add(tf.keras.layers.Dense(64, activation = 'relu'))  # Mantenha o número de unidades
lModeloDNN4.add(tf.keras.layers.Dense(len(set(lLinguas)), activation = 'softmax'))

# Configuração personalizada do otimizador com taxa de aprendizado
lOtimizador = tf.keras.optimizers.Adam(learning_rate = 0.001)
lModeloDNN4.compile(loss = 'sparse_categorical_crossentropy', optimizer = lOtimizador, metrics = ['accuracy'])

# Treinando e avaliando o modelo
lHistoricoDNNAprendizado = lModeloDNN4.fit(lSequenciasTreino, lLinguasTreinoCodificado, epochs=500, batch_size=1, validation_data=(lSequenciasTeste, lLinguasTesteCodificado))
lPredicaoY = lModeloDNN4.predict(lSequenciasTeste)
lPredicaoClassesY = np.argmax(lPredicaoY, axis=1)
lAcuracia = accuracy_score(lLinguasTesteCodificado, lPredicaoClassesY)
print(f"Acurácia do modelo lModeloDNN4: {lAcuracia}")
Helper.MostrarPerdaPlot(lHistoricoDNNAprendizado, 'DNN Rate Learn Model')
lClassesUnique = np.unique(lLinguas)
lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoClassesY)
Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'DNN Rate Learn Model')

# Salvando o modelo
lModeloDNN4.save(Helper.DiretorioAtual() + '/Models/lModeloDNN4.keras')


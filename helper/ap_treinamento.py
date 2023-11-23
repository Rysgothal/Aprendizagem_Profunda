import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

try:
    from helper.ap_helper import Helper 
except:
    from ap_helper import Helper

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

# def Helper.MostrarPerdaPlot(history, model_name):
#     plt.plot(history.history['loss'], label='train')
#     plt.plot(history.history['val_loss'], label='validation')
#     plt.title(f'{model_name} - Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()
    
# def Helper.MostrarMatrixConfusao(cm, classes, model_name):
#     cm_df = pd.DataFrame(cm, index=classes[:len(cm)], columns=classes[:len(cm)])
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm_df, annot=True, cmap='Blues')  
#     plt.title(f'{model_name} - Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.show()

    
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
filter_sizes_list = [3, 5, 7]

for filter_size in filter_sizes_list:
    # Crie um modelo CNN com a variação do tamanho do filtro
    model_filter_size = tf.keras.Sequential()
    model_filter_size.add(tf.keras.layers.Embedding(input_dim=lTamanhoVocabulario, output_dim=32, input_length=len(lSequencias[0])))
    model_filter_size.add(tf.keras.layers.Conv1D(128, filter_size, activation='relu'))  # Varie o tamanho do filtro conforme desejado
    model_filter_size.add(tf.keras.layers.GlobalMaxPooling1D())
    model_filter_size.add(tf.keras.layers.Dense(len(set(lLinguas)), activation='softmax'))
    model_filter_size.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Treine o modelo
    history_cnn_filter_size=model_filter_size.fit(lSequenciasTreino, lLinguasTreinoCodificado, epochs=500, batch_size=1, validation_data=(lSequenciasTeste, lLinguasTesteCodificado))

    # Salve o modelo com um nome indicando a variação do hiperparâmetro
    model_filter_size.save(Helper.DiretorioAtual() + f'/Models/modelCNN_filter_size_{filter_size}.keras')

    # Avaliação do modelo
    lPredicaoY = model_filter_size.predict(lSequenciasTeste)
    lPredicaoClassesY = np.argmax(lPredicaoY, axis=1)
    lAcuracia = accuracy_score(lLinguasTesteCodificado, lPredicaoClassesY)
    print(f"Acurácia do modelo CNN (filtro de tamanho {filter_size}): {lAcuracia}")
    Helper.MostrarPerdaPlot(history_cnn_filter_size, 'CNN Filter Size Model')
    lClassesUnique = np.unique(lLinguas)
    lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoClassesY)
    Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'CNN Filter Size Model')
    

# Modelo 2 com tamanho do passo diferente
filter_stride_list = [1, 2, 3]

for filter_stride in filter_stride_list:
    # Crie um modelo CNN com a variação do tamanho do passo
    model_filter_stride = tf.keras.Sequential()
    model_filter_stride.add(tf.keras.layers.Embedding(input_dim=lTamanhoVocabulario, output_dim=32, input_length=len(lSequencias[0])))
    model_filter_stride.add(tf.keras.layers.Conv1D(128, 5, strides=filter_stride, activation='relu'))  # Varie o tamanho do passo conforme desejado
    model_filter_stride.add(tf.keras.layers.GlobalMaxPooling1D())
    model_filter_stride.add(tf.keras.layers.Dense(len(set(lLinguas)), activation='softmax'))
    model_filter_stride.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Treine o modelo
    history_cnn_filter_stride=model_filter_stride.fit(lSequenciasTreino, lLinguasTreinoCodificado, epochs=500, batch_size=1, validation_data=(lSequenciasTeste, lLinguasTesteCodificado))

    # Salve o modelo com um nome indicando a variação do hiperparâmetro
    model_filter_stride.save(Helper.DiretorioAtual() + f'/Models/modelCNN_filter_stride_{filter_stride}.keras')

    # Avaliação do modelo
    lPredicaoY = model_filter_stride.predict(lSequenciasTeste)
    lPredicaoClassesY = np.argmax(lPredicaoY, axis=1)
    lAcuracia = accuracy_score(lLinguasTesteCodificado, lPredicaoClassesY)
    print(f"Acurácia do modelo CNN (passo de tamanho {filter_stride}): {lAcuracia}")
    Helper.MostrarPerdaPlot(history_cnn_filter_stride,'CNN Filter Stride Model')
    lClassesUnique = np.unique(lLinguas)
    lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoClassesY)
    Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'CNN Filter Stride Model')


unique, counts = np.unique(lLinguasTesteCodificado, return_counts=True)
print("Distribuição das classes no conjunto de teste:")
print(dict(zip(unique, counts)))
# Modelo DNNs (Rede Neural Profunda) com número de camadas ocultas
model_layers = tf.keras.Sequential(name="model_layers")
model_layers.add(tf.keras.layers.Input(shape=(len(lSequencias[0]),)))  # Altere o tamanho da sequência de acordo
model_layers.add(tf.keras.layers.Dense(64, activation='relu'))
model_layers.add(tf.keras.layers.Dense(64, activation='relu'))
model_layers.add(tf.keras.layers.Dense(64, activation='relu'))
model_layers.add(tf.keras.layers.Dense(len(set(lLinguas)), activation='softmax'))
model_layers.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Treine e avalie o modelo model_layers
history_dnn_ocult=model_layers.fit(lSequenciasTreino, lLinguasTreinoCodificado, epochs=500, batch_size=1, validation_data=(lSequenciasTeste, lLinguasTesteCodificado))
y_pred_layers = model_layers.predict(lSequenciasTeste)
y_pred_classes_layers = np.argmax(y_pred_layers, axis=1)
accuracy_layers = accuracy_score(lLinguasTesteCodificado, y_pred_classes_layers)
print(f"Acurácia do modelo model_layers: {accuracy_layers}")
Helper.MostrarPerdaPlot(history_dnn_ocult, 'DNN Ocult Model')
lClassesUnique = np.unique(lLinguas)
lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, y_pred_classes_layers)
Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'DNN Ocult Model')
# Salve o modelo model_layers no diretório atual
model_layers.save(Helper.DiretorioAtual() + '/Models/modelDNN_layers.keras')


# Modelo DNNs (Rede Neural Profunda) com diferentes funções de ativação nas camadas ocultas
modelDNN = tf.keras.Sequential(name="modelDNN")
modelDNN.add(tf.keras.layers.Input(shape=(len(lSequencias[0]),)))  # Altere o tamanho da sequência de acordo
modelDNN.add(tf.keras.layers.Dense(64, activation='relu'))
modelDNN.add(tf.keras.layers.Dense(64, activation='tanh'))  # Varie a função de ativação conforme desejado
modelDNN.add(tf.keras.layers.Dense(64, activation='sigmoid'))
modelDNN.add(tf.keras.layers.Dense(len(set(lLinguas)), activation='softmax'))
modelDNN.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Treine e avalie o modelo modelDNN
history_dnn_ocult_ativation=modelDNN.fit(lSequenciasTreino, lLinguasTreinoCodificado, epochs=500, batch_size=1, validation_data=(lSequenciasTeste, lLinguasTesteCodificado))

y_pred_dnn = modelDNN.predict(lSequenciasTeste)
y_pred_classes_dnn = np.argmax(y_pred_dnn, axis=1)
accuracy_dnn = accuracy_score(lLinguasTesteCodificado, y_pred_classes_dnn)
print(f"Acurácia do modelo modelDNN: {accuracy_dnn}")
Helper.MostrarPerdaPlot(history_dnn_ocult_ativation, 'DNN Ocult Ativation Model')
lClassesUnique = np.unique(lLinguas)
lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, y_pred_classes_dnn)
Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'DNN Ocult Ativation Model')

# Salve o modelo modelDNN
modelDNN.save(Helper.DiretorioAtual() + '/Models/modelDNN.keras')


# Modelo DNNs (Rede Neural Profunda) com variação da taxa de aprendizado e função de ativação
modelDNN4 = tf.keras.Sequential(name="modelDNN4")
modelDNN4.add(tf.keras.layers.Input(shape=(len(lSequencias[0]),)))  # Ajuste o tamanho da sequência de acordo
modelDNN4.add(tf.keras.layers.Dense(64, activation='relu'))  # Mantenha o número de unidades
modelDNN4.add(tf.keras.layers.Dense(64, activation='relu'))  # Mantenha o número de unidades
modelDNN4.add(tf.keras.layers.Dense(len(set(lLinguas)), activation='softmax'))

# Configuração personalizada do otimizador com taxa de aprendizado
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
modelDNN4.compile(loss='sparse_categorical_crossentropy', optimizer=custom_optimizer, metrics=['accuracy'])

# Treine e avalie o modelo
history_dnn_rate_learn=modelDNN4.fit(lSequenciasTreino, lLinguasTreinoCodificado, epochs=500, batch_size=1, validation_data=(lSequenciasTeste, lLinguasTesteCodificado))
lPredicaoY = modelDNN4.predict(lSequenciasTeste)
lPredicaoClassesY = np.argmax(lPredicaoY, axis=1)
lAcuracia = accuracy_score(lLinguasTesteCodificado, lPredicaoClassesY)
print(f"Acurácia do modelo modelDNN4: {lAcuracia}")
Helper.MostrarPerdaPlot(history_dnn_rate_learn, 'DNN Rate Learn Model')
lClassesUnique = np.unique(lLinguas)
lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoClassesY)
Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'DNN Rate Learn Model')

# Salve o modelo
modelDNN4.save(Helper.DiretorioAtual() + '/Models/modelDNN4.keras')


import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score,confusion_matrix

def CarregarLinguagemDataSet(pCaminho):
    lFrases = []
    lLinguas = []

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
                lLinguas.append(lCategoria)
    
    return lFrases, lLinguas

lCaminhoRaiz = os.path.abspath(os.path.dirname(__file__))
lCaminhoTreino = lCaminhoRaiz + '/Textos/train'
lCaminhoTeste = lCaminhoRaiz + '/Textos/test'

lTreinoFrases, lTreinoLinguas = CarregarLinguagemDataSet(lCaminhoTreino) # Carregando o dataset de treinamento
lTesteTextos, lTesteLinguas = CarregarLinguagemDataSet(lCaminhoTeste)    # Carregando o dataset de teste

# Combinando os textos e rótulos do treinamento e teste
lTextos = lTreinoFrases + lTesteTextos
lLinguas = lTreinoLinguas + lTesteLinguas

# Pré-processamento de texto
lTokenizacaoTexto = tf.keras.layers.TextVectorization(max_tokens = 20000, output_mode = 'int')
lTokenizacaoTexto.adapt(lTextos)

lSequencias = lTokenizacaoTexto(lTextos)
# Pré-processamento dos dados de teste
lSequenciasTreino = lTokenizacaoTexto(np.array(lTreinoFrases)).numpy()
lSequenciasTeste = lTokenizacaoTexto(np.array(lTesteTextos)).numpy()
lTamanhoMaximo = max(len(seq) for seq in lSequenciasTreino)

lSequenciasTreino = tf.keras.preprocessing.sequence.pad_sequences(lSequenciasTreino, maxlen = lTamanhoMaximo, padding='post')
lSequenciasTeste = tf.keras.preprocessing.sequence.pad_sequences(lSequenciasTeste, maxlen = lTamanhoMaximo, padding='post')

# Tamanho do vocabulário
lTamanhoVocabulario = len(lTokenizacaoTexto.get_vocabulary())

# Codificação das etiquetas
lLiguaCodificador = LabelEncoder()
lLiguaCodificador.fit(lTreinoLinguas)
lLinguasTesteCodificado = lLiguaCodificador.transform(lTesteLinguas)
lLinguasTreinoEncodificado = lLiguaCodificador.transform(lTreinoLinguas)
lLinguasEncodificado = lLiguaCodificador.fit_transform(lLinguas)

lSequenciasTreino = lTokenizacaoTexto(lTreinoFrases)
lTamanhoMaximo = max(len(seq) for seq in lSequenciasTreino)

lSequenciasTreino = tf.keras.preprocessing.sequence.pad_sequences(lSequenciasTreino, maxlen=lTamanhoMaximo, padding='post')
lSequenciasTeste = tf.keras.preprocessing.sequence.pad_sequences(lSequenciasTeste, maxlen=lTamanhoMaximo, padding='post')

# Converte o tensor sequences em um array NumPy
lSequencias = lSequencias.numpy()

def MostrarPerdaPlot(pHistorico, pNomeModelo):
    plt.plot(pHistorico.history['loss'], label = 'Treino')
    plt.plot(pHistorico.history['val_loss'], label = 'Validação')
    plt.title(f'{pNomeModelo} - Perda')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.show()
    
def MostrarMatrizConfusãoPlot(pMedidaCM, pClasses, pNomeModelo):
    cm_df = pd.DataFrame(pMedidaCM, index = pClasses[:len(pMedidaCM)], columns = pClasses[:len(pMedidaCM)])
    
    plt.figure(figsize = (10, 8))
    sns.heatmap(cm_df, annot = True, fmt = 'g', cmap = 'Blues')
    plt.title(f'{pNomeModelo} - Confusion Matrix')
    plt.xlabel('Predição')
    plt.ylabel('Lingua')
    plt.show()

    
# Modelo ANNs (Redes Neurais Artificiais) com dropout

model_dropout = tf.keras.Sequential()
model_dropout.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=len(sequences[0])))
model_dropout.add(tf.keras.layers.GlobalAveragePooling1D())  # Camada de pooling
model_dropout.add(tf.keras.layers.Dense(64, activation='relu'))
model_dropout.add(tf.keras.layers.Dropout(0.5))
model_dropout.add(tf.keras.layers.Dense(len(set(lLinguas)), activation='softmax'))
model_dropout.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treine o modelo usando as sequências de treinamento e rótulos codificados
history_dropout = model_dropout.fit(sequences_train, lLinguas_train_encoded, epochs=500, batch_size=1, validation_data=(sequences_test, lLinguas_test_encoded))

# Avaliação do modelo usando os dados de teste
y_pred_dropout = model_dropout.predict(sequences_test)
y_pred_classes_dropout = np.argmax(y_pred_dropout, axis=1)
accuracy_dropout = accuracy_score(lLinguas_test_encoded, y_pred_classes_dropout)
print(f"Acurácia do modelo ANNs com dropout: {accuracy_dropout}")

# Plote o gráfico de perda e a matriz de confusão
MostrarPerda(history_dropout, 'ANN Dropout Model')
# Exemplo de uso
unique_classes = np.unique(lLinguas)
print("classes:", unique_classes)
confusion_mat = confusion_matrix(lLinguas_test_encoded, y_pred_classes_dropout, normalize='true')
print("matriz:", confusion_mat)
MostrarMatrizConfusãoPlot(confusion_mat, classes=unique_classes, model_name='ANN Dropout Model')

model_dropout.save('modelANNs_dropout.keras')


# Variações de hiperparâmetros para o modelo ANNs
ann_units_list = [32, 64, 128]

# Crie, treine e avalie um modelo ANNs para cada variação de unidades ANNs
for ann_units in ann_units_list:
    # Crie um modelo ANNs com a variação do número de unidades ANNs
    model_ann = tf.keras.Sequential()
    model_ann.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=len(sequences[0])))
    model_ann.add(tf.keras.layers.GlobalAveragePooling1D())  # Camada de pooling
    model_ann.add(tf.keras.layers.Dense(ann_units, activation='relu'))
    model_ann.add(tf.keras.layers.Dense(len(set(lLinguas)), activation='softmax'))
    model_ann.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Treine o modelo
    history_ann=model_ann.fit(X_train, y_train, epochs=500, batch_size=1, validation_data=(X_test, y_test))
    
    y_pred = model_ann.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"Acurácia do modelo ANNs ({ann_units}): {accuracy}")
    # Plote o gráfico de perda e a matriz de confusão
    MostrarPerda(history_ann, 'ANN Model')
    unique_classes = np.unique(lLinguas)
    confusion_mat = confusion_matrix(y_test, y_pred_classes)
    plot_confusion_matrix(confusion_mat, classes=unique_classes, model_name='ANN Model')
    # Salve o modelo com um nome indicando a variação do hiperparâmetro
    model_ann.save(f'modelANNs_{ann_units}.keras')


# Terceiro Modelo ANNs com Camada LSTM
model_lstm = tf.keras.Sequential()
model_lstm.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=len(sequences[0])))
model_lstm.add(tf.keras.layers.LSTM(64, return_sequences=True))  # Camada LSTM
model_lstm.add(tf.keras.layers.GlobalMaxPooling1D())  # Camada de pooling
model_lstm.add(tf.keras.layers.Dense(len(set(lLinguas)), activation='softmax'))
model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treine o modelo LSTM
history_ann_lstm=model_lstm.fit(X_train, y_train, epochs=500, batch_size=1, validation_data=(X_test, y_test))

# Avaliação do modelo LSTM
y_pred_lstm = model_lstm.predict(X_test)
y_pred_classes_lstm = np.argmax(y_pred_lstm, axis=1)
accuracy_lstm = accuracy_score(y_test, y_pred_classes_lstm)
print(f"Acurácia do modelo ANNs com LSTM: {accuracy_lstm}")
MostrarPerda(history_ann_lstm, 'ANN LSTM Model')
unique_classes = np.unique(lLinguas)
confusion_mat = confusion_matrix(y_test, y_pred_classes_lstm)
plot_confusion_matrix(confusion_mat, classes=unique_classes, model_name='ANN LSTM Model')
model_lstm.save('modelANNs_lstm.keras')

# Variações de hiperparâmetros para o modelo CNN
num_filters_list = [64, 128, 256]

# Crie, treine e avalie um modelo CNN para cada variação de número de filtros
for num_filters in num_filters_list:
    # Crie um modelo CNN com a variação do número de filtros
    model_cnn = tf.keras.Sequential()
    model_cnn.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=len(sequences[0])))
    model_cnn.add(tf.keras.layers.Conv1D(num_filters, 5, activation='relu'))
    model_cnn.add(tf.keras.layers.GlobalMaxPooling1D())
    model_cnn.add(tf.keras.layers.Dense(len(set(lLinguas)), activation='softmax'))
    model_cnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Treine o modelo
    history_cnn_filter=model_cnn.fit(X_train, y_train, epochs=500, batch_size=1, validation_data=(X_test, y_test))

    # Avaliação do modelo
    y_pred = model_cnn.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"Acurácia do modelo CNN ({num_filters} filtros): {accuracy}")
    MostrarPerda(history_cnn_filter, 'CNN Filter Model')
    unique_classes = np.unique(lLinguas)
    confusion_mat = confusion_matrix(y_test, y_pred_classes)
    plot_confusion_matrix(confusion_mat, classes=unique_classes, model_name='CNN Filter Model')
    
    # Salve o modelo com um nome indicando a variação do hiperparâmetro
    model_cnn.save(f'modelCNN_{num_filters}.keras')

# Variação de outro hiperparâmetro para os modelos CNN
# Modelo 1 com tamanho de filtro diferente
filter_sizes_list = [3, 5, 7]

for filter_size in filter_sizes_list:
    # Crie um modelo CNN com a variação do tamanho do filtro
    model_filter_size = tf.keras.Sequential()
    model_filter_size.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=len(sequences[0])))
    model_filter_size.add(tf.keras.layers.Conv1D(128, filter_size, activation='relu'))  # Varie o tamanho do filtro conforme desejado
    model_filter_size.add(tf.keras.layers.GlobalMaxPooling1D())
    model_filter_size.add(tf.keras.layers.Dense(len(set(lLinguas)), activation='softmax'))
    model_filter_size.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Treine o modelo
    history_cnn_filter_size=model_filter_size.fit(X_train, y_train, epochs=500, batch_size=1, validation_data=(X_test, y_test))

    # Salve o modelo com um nome indicando a variação do hiperparâmetro
    model_filter_size.save(f'modelCNN_filter_size_{filter_size}.keras')

    # Avaliação do modelo
    y_pred = model_filter_size.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"Acurácia do modelo CNN (filtro de tamanho {filter_size}): {accuracy}")
    MostrarPerda(history_cnn_filter_size, 'CNN Filter Size Model')
    unique_classes = np.unique(lLinguas)
    confusion_mat = confusion_matrix(y_test, y_pred_classes)
    plot_confusion_matrix(confusion_mat, classes=unique_classes, model_name='CNN Filter Size Model')
    

# Modelo 2 com tamanho do passo diferente
filter_stride_list = [1, 2, 3]

for filter_stride in filter_stride_list:
    # Crie um modelo CNN com a variação do tamanho do passo
    model_filter_stride = tf.keras.Sequential()
    model_filter_stride.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=len(sequences[0])))
    model_filter_stride.add(tf.keras.layers.Conv1D(128, 5, strides=filter_stride, activation='relu'))  # Varie o tamanho do passo conforme desejado
    model_filter_stride.add(tf.keras.layers.GlobalMaxPooling1D())
    model_filter_stride.add(tf.keras.layers.Dense(len(set(lLinguas)), activation='softmax'))
    model_filter_stride.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Treine o modelo
    history_cnn_filter_stride=model_filter_stride.fit(X_train, y_train, epochs=500, batch_size=1, validation_data=(X_test, y_test))

    # Salve o modelo com um nome indicando a variação do hiperparâmetro
    model_filter_stride.save(f'modelCNN_filter_stride_{filter_stride}.keras')

    # Avaliação do modelo
    y_pred = model_filter_stride.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"Acurácia do modelo CNN (passo de tamanho {filter_stride}): {accuracy}")
    MostrarPerda(history_cnn_filter_stride,'CNN Filter Stride Model')
    unique_classes = np.unique(lLinguas)
    confusion_mat = confusion_matrix(y_test, y_pred_classes)
    plot_confusion_matrix(confusion_mat, classes=unique_classes, model_name='CNN Filter Stride Model')


unique, counts = np.unique(y_test, return_counts=True)
print("Distribuição das classes no conjunto de teste:")
print(dict(zip(unique, counts)))
# Modelo DNNs (Rede Neural Profunda) com número de camadas ocultas
model_layers = tf.keras.Sequential(name="model_layers")
model_layers.add(tf.keras.layers.Input(shape=(len(sequences[0]),)))  # Altere o tamanho da sequência de acordo
model_layers.add(tf.keras.layers.Dense(64, activation='relu'))
model_layers.add(tf.keras.layers.Dense(64, activation='relu'))
model_layers.add(tf.keras.layers.Dense(64, activation='relu'))
model_layers.add(tf.keras.layers.Dense(len(set(lLinguas)), activation='softmax'))
model_layers.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Treine e avalie o modelo model_layers
history_dnn_ocult=model_layers.fit(X_train, y_train, epochs=500, batch_size=1, validation_data=(X_test, y_test))
y_pred_layers = model_layers.predict(X_test)
y_pred_classes_layers = np.argmax(y_pred_layers, axis=1)
accuracy_layers = accuracy_score(y_test, y_pred_classes_layers)
print(f"Acurácia do modelo model_layers: {accuracy_layers}")
MostrarPerda(history_dnn_ocult, 'DNN Ocult Model')
unique_classes = np.unique(lLinguas)
confusion_mat = confusion_matrix(y_test, y_pred_classes_layers)
plot_confusion_matrix(confusion_mat, classes=unique_classes, model_name='DNN Ocult Model')
# Salve o modelo model_layers no diretório atual
model_layers.save('modelDNN_layers.keras')


# Modelo DNNs (Rede Neural Profunda) com diferentes funções de ativação nas camadas ocultas
modelDNN = tf.keras.Sequential(name="modelDNN")
modelDNN.add(tf.keras.layers.Input(shape=(len(sequences[0]),)))  # Altere o tamanho da sequência de acordo
modelDNN.add(tf.keras.layers.Dense(64, activation='relu'))
modelDNN.add(tf.keras.layers.Dense(64, activation='tanh'))  # Varie a função de ativação conforme desejado
modelDNN.add(tf.keras.layers.Dense(64, activation='sigmoid'))
modelDNN.add(tf.keras.layers.Dense(len(set(lLinguas)), activation='softmax'))
modelDNN.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Treine e avalie o modelo modelDNN
history_dnn_ocult_ativation=modelDNN.fit(X_train, y_train, epochs=500, batch_size=1, validation_data=(X_test, y_test))

y_pred_dnn = modelDNN.predict(X_test)
y_pred_classes_dnn = np.argmax(y_pred_dnn, axis=1)
accuracy_dnn = accuracy_score(y_test, y_pred_classes_dnn)
print(f"Acurácia do modelo modelDNN: {accuracy_dnn}")
MostrarPerda(history_dnn_ocult_ativation, 'DNN Ocult Ativation Model')
unique_classes = np.unique(lLinguas)
confusion_mat = confusion_matrix(y_test, y_pred_classes_dnn)
plot_confusion_matrix(confusion_mat, classes=unique_classes, model_name='DNN Ocult Ativation Model')

# Salve o modelo modelDNN
modelDNN.save('modelDNN.keras')


# Modelo DNNs (Rede Neural Profunda) com variação da taxa de aprendizado e função de ativação
modelDNN4 = tf.keras.Sequential(name="modelDNN4")
modelDNN4.add(tf.keras.layers.Input(shape=(len(sequences[0]),)))  # Ajuste o tamanho da sequência de acordo
modelDNN4.add(tf.keras.layers.Dense(64, activation='relu'))  # Mantenha o número de unidades
modelDNN4.add(tf.keras.layers.Dense(64, activation='relu'))  # Mantenha o número de unidades
modelDNN4.add(tf.keras.layers.Dense(len(set(lLinguas)), activation='softmax'))

# Configuração personalizada do otimizador com taxa de aprendizado
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
modelDNN4.compile(loss='sparse_categorical_crossentropy', optimizer=custom_optimizer, metrics=['accuracy'])

# Treine e avalie o modelo
history_dnn_rate_learn=modelDNN4.fit(X_train, y_train, epochs=500, batch_size=1, validation_data=(X_test, y_test))
y_pred = modelDNN4.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Acurácia do modelo modelDNN4: {accuracy}")
MostrarPerda(history_dnn_rate_learn, 'DNN Rate Learn Model')
unique_classes = np.unique(lLinguas)
confusion_mat = confusion_matrix(y_test, y_pred_classes)
plot_confusion_matrix(confusion_mat, classes=unique_classes, model_name='DNN Rate Learn Model')

# Salve o modelo
modelDNN4.save('modelDNN4.keras')




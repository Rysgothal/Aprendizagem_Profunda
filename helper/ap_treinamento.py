import os
import tensorflow as tf

# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score,confusion_matrix
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt

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
##asdasdas Parei aQ

sequences = lTokenizacaoTexto(lTextos)
# Pré-processamento dos dados de teste
sequences_train = lTokenizacaoTexto(np.array(lTreinoFrases)).numpy()
sequences_test = lTokenizacaoTexto(np.array(lTesteTextos)).numpy()
max_length = max(len(seq) for seq in sequences_train)
sequences_train = tf.keras.preprocessing.sequence.pad_sequences(sequences_train, maxlen=max_length, padding='post')
sequences_test = tf.keras.preprocessing.sequence.pad_sequences(sequences_test, maxlen=max_length, padding='post')
# Tamanho do vocabulário
vocab_size = len(lTokenizacaoTexto.get_vocabulary())
# Codificação das etiquetas
label_encoder = LabelEncoder()
label_encoder.fit(lTreinoLinguas)
lLinguas_test_encoded = label_encoder.transform(lTesteLinguas)
lLinguas_encoded = label_encoder.fit_transform(lLinguas)
lLinguas_train_encoded = label_encoder.transform(lTreinoLinguas)
sequences_train = lTokenizacaoTexto(lTreinoFrases)
max_length = max(len(seq) for seq in sequences_train)

sequences_train = tf.keras.preprocessing.sequence.pad_sequences(sequences_train, maxlen=max_length, padding='post')
sequences_test = tf.keras.preprocessing.sequence.pad_sequences(sequences_test, maxlen=max_length, padding='post')

# Converte o tensor sequences em um array NumPy
sequences = sequences.numpy()

# # Divisão dos dados em treinamento, validação e teste como arrays NumPy
# X_train, X_temp, y_train, y_temp = train_test_split(sequences, lLinguas_encoded, test_size=0.2, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(sequences_train.numpy(), lLinguas_train_encoded, test_size=0.6, random_state=1)


def plot_loss(history, model_name):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def plot_confusion_matrix(cm, classes, model_name):
    cm_df = pd.DataFrame(cm, index=classes[:len(cm)], columns=classes[:len(cm)])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
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
plot_loss(history_dropout, 'ANN Dropout Model')
# Exemplo de uso
unique_classes = np.unique(lLinguas)
print("classes:", unique_classes)
confusion_mat = confusion_matrix(lLinguas_test_encoded, y_pred_classes_dropout, normalize='true')
print("matriz:", confusion_mat)
plot_confusion_matrix(confusion_mat, classes=unique_classes, model_name='ANN Dropout Model')

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
    plot_loss(history_ann, 'ANN Model')
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
plot_loss(history_ann_lstm, 'ANN LSTM Model')
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
    plot_loss(history_cnn_filter, 'CNN Filter Model')
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
    plot_loss(history_cnn_filter_size, 'CNN Filter Size Model')
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
    plot_loss(history_cnn_filter_stride,'CNN Filter Stride Model')
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
plot_loss(history_dnn_ocult, 'DNN Ocult Model')
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
plot_loss(history_dnn_ocult_ativation, 'DNN Ocult Ativation Model')
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
plot_loss(history_dnn_rate_learn, 'DNN Rate Learn Model')
unique_classes = np.unique(lLinguas)
confusion_mat = confusion_matrix(y_test, y_pred_classes)
plot_confusion_matrix(confusion_mat, classes=unique_classes, model_name='DNN Rate Learn Model')

# Salve o modelo
modelDNN4.save('modelDNN4.keras')




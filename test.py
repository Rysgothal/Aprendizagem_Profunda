import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Treinamento import *
# Carregar o conjunto de dados de teste (substitua 'dataset_teste.txt' pelo caminho do arquivo de teste real)
file_path_test = "dataset_test.txt"
texts_test, labels_test = load_language_dataset(file_path_test)

# Pré-processamento do conjunto de dados de teste
sequences_test = tokenizer(texts_test)
sequences_test = sequences_test.numpy()
labels_encoded_test = label_encoder.transform(labels_test)

# Inicialize as listas para o ranking e as matrizes de confusão
accuracy_ranking = []
confusion_matrices = []

# Avaliar os modelos
for model, model_name in zip(models, model_names):
    # Descubra o tamanho máximo da sequência entre todos os modelos
    max_input_size = max(model.layers[0].input_shape[1] for model in models)
    sequences_test_padded = pad_sequences(sequences_test, maxlen=max_input_size, padding='post', truncating='post')

    y_pred = model.predict(sequences_test_padded)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(labels_encoded_test, y_pred_classes)
    accuracy_ranking.append((model_name, accuracy))
    
    # Calcular a matriz de confusão
    cm = confusion_matrix(labels_encoded_test, y_pred_classes)
    confusion_matrices.append((model_name, cm))

# Classificar os modelos por acurácia
accuracy_ranking.sort(key=lambda x: x[1], reverse=True)

# Imprimir o ranking de acurácia com atraso
for i, (model_name, accuracy) in enumerate(accuracy_ranking, start=1):
    print(f"Rank {i}: Modelo {model_name} - Acurácia: {accuracy:.4f}")
    time.sleep(1)  # Adicione um atraso de 1 segundo

# Imprimir as matrizes de confusão com atraso
for model_name, cm in confusion_matrices:
    print(f"Matriz de Confusão para o Modelo {model_name}:")
    print(cm)
    time.sleep(1)  # Adicione um atraso de 1 segundo

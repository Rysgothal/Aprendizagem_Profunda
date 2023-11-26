import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import os
from helper.ap_helper import Helper, Modelo 

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
lANNDropout = Modelo('Modelo_ANN_Dropout')
lANNDropout.AdicionarEmbedding(lTamanhoVocabulario, len(lSequencias[0]))
lANNDropout.AdicionarPooling()
lANNDropout.AdicionarDense()
lANNDropout.AdicionarDropOut(0.5)
lANNDropout.AdicionarDense(len(set(lLinguas)), 'softmax') 
lANNDropout.Compilar('sparse_categorical_crossentropy', 'adam', ['accuracy'])

# Treinando o modelo
lHistoricoDropOutANN = lANNDropout.Treinamento(lSequenciasTreino, lLinguasTreinoCodificado, 
                                              (lSequenciasTeste, lLinguasTesteCodificado))

# Avaliação do modelo usando os dados de teste
lDropOutPredicaoY = lANNDropout.Modelo.predict(lSequenciasTeste)
lDropOutPredicaoClassesY = np.argmax(lDropOutPredicaoY, axis = 1)
lDropOutAcuracia = accuracy_score(lLinguasTesteCodificado, lDropOutPredicaoClassesY)
print(f"Acurácia do modelo ANNs com dropout: {lDropOutAcuracia}")

# Plotando o gráfico de perda e a matriz de confusão
Helper.MostrarPerdaPlot(lHistoricoDropOutANN, 'ANN Dropout Model')

lClassesUnique = np.unique(lLinguas)
print("Classes: ", lClassesUnique)
lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lDropOutPredicaoClassesY, normalize = 'true')
print("Matriz:", lMatrizConfusao)
Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'ANN Dropout Model')

lANNDropout.Salvar('modelANNs_dropout')

# Variações de hiperparâmetros para o modelo ANNs
lUnidadesANN = [32, 64, 128]

# Criando, treinando e avaliando modelo ANNs para cada variação de unidades ANNs
for lUnidadeANN in lUnidadesANN:
    # Criando um modelo ANNs com a variação do número de unidades ANNs
    lModeloANN = Modelo('Modelo_ANN_Numero_Neuronios')
    lModeloANN.AdicionarEmbedding(lTamanhoVocabulario, len(lSequencias[0]))
    lModeloANN.AdicionarPooling()
    lModeloANN.AdicionarDense(lUnidadeANN)
    lModeloANN.AdicionarDense(len(set(lLinguas)), 'softmax')
    lModeloANN.Compilar('sparse_categorical_crossentropy', 'adam', ['accuracy'])
    
    # Treinando o modelo
    lHistoricoANN = lModeloANN.Treinamento(lSequenciasTreino, lLinguasTreinoCodificado,
                                          (lSequenciasTeste, lLinguasTesteCodificado))
    
    lPredicaoY = lModeloANN.Modelo.predict(lSequenciasTeste)
    lPredicaoClassesY = np.argmax(lPredicaoY, axis = 1)
    lAcuracia = accuracy_score(lLinguasTesteCodificado, lPredicaoClassesY)

    print(f"Acurácia do modelo ANNs ({lUnidadeANN}): {lAcuracia}")
    Helper.MostrarPerdaPlot(lHistoricoANN, 'ANN Model')

    lClassesUnique = np.unique(lLinguas)
    lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoClassesY)
    Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'ANN Model')

    lModeloANN.Salvar(f'modelANNs_{lUnidadeANN}')


# Terceiro Modelo ANNs com Camada LSTM
lANNCamadaLSTM = Modelo('Modelo_ANN_LSTM')
lANNCamadaLSTM.AdicionarEmbedding(lTamanhoVocabulario, len(lSequencias[0]))
lANNCamadaLSTM.AdicionarLSTM(64, True)
lANNCamadaLSTM.AdicionarPooling()
lANNCamadaLSTM.AdicionarDense(len(set(lLinguas)), 'softmax')
lANNCamadaLSTM.Compilar('sparse_categorical_crossentropy', 'adam', ['accuracy'])

# Treinando modelo
lHistoricoANNCamadaLSTM = lANNCamadaLSTM.Treinamento(lSequenciasTreino, lLinguasTreinoCodificado,
                                                    (lSequenciasTeste, lLinguasTesteCodificado))

# Avaliando do modelo LSTM
lPredicaoYCamadaLSTM = lANNCamadaLSTM.Modelo.predict(lSequenciasTeste)
lPredicaoYCamadaLSTMClasses = np.argmax(lPredicaoYCamadaLSTM, axis = 1)

lAcuraciaLSTM = accuracy_score(lLinguasTesteCodificado, lPredicaoYCamadaLSTMClasses)
print(f"Acurácia do modelo ANNs com LSTM: {lAcuraciaLSTM}")
Helper.MostrarPerdaPlot(lHistoricoANNCamadaLSTM, 'ANN LSTM Model')

lClassesUnique = np.unique(lLinguas)
lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoYCamadaLSTMClasses)
Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'ANN LSTM Model')

lANNCamadaLSTM.Salvar(f'modelANNs_lstm')

# Variações de hiperparâmetros para o modelo CNN
lNumeroFiltrosLista = [64, 128, 256]

# Criando, treinando e avaliando um modelo CNN para cada variação de número de filtros
for lNumerosFitros in lNumeroFiltrosLista:
    # Criando um modelo CNN com a variação do número de filtros
    lModeloCNN = Modelo('Modelo_CNN_Numero_Filtros')
    lModeloCNN.AdicionarEmbedding(lTamanhoVocabulario, len(lSequencias[0]))
    lModeloCNN.AdicionarConvUnidirecional(pNumFiltros = lNumerosFitros, pTamanhoConv = 5, pAtivacao = 'relu')
    lModeloCNN.AdicionarPooling()
    lModeloCNN.AdicionarDense(len(set(lLinguas)), 'softmax')
    lModeloCNN.Compilar('sparse_categorical_crossentropy', 'adam', ['accuracy'])
    
    # Treinando o modelo
    lHistoricoCNNFiltro = lModeloCNN.Treinamento(lSequenciasTreino, lLinguasTreinoCodificado, 
                                                 (lSequenciasTeste, lLinguasTesteCodificado))

    # Avaliando do modelo
    lPredicaoY = lModeloCNN.Modelo.predict(lSequenciasTeste)
    lPredicaoClassesY = np.argmax(lPredicaoY, axis=1)
    
    lAcuracia = accuracy_score(lLinguasTesteCodificado, lPredicaoClassesY)
    print(f"Acurácia do modelo CNN ({lNumerosFitros} filtros): {lAcuracia}")
    Helper.MostrarPerdaPlot(lHistoricoCNNFiltro ,  'CNN Filter Model')
    
    lClassesUnique = np.unique(lLinguas)
    lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoClassesY)
    Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'CNN Filter Model')
    
    lModeloCNN.Salvar(f'modelCNN_{lNumerosFitros}')

# Modelos com tamanhos de filtros diferentes
lTamanhoFiltroLista = [3, 5, 7] 

for lTamanhoFiltro in lTamanhoFiltroLista:
    # Criando um modelo CNN com a variação do tamanho do filtro
    lCNNTamanhoFiltro = Modelo('Modelo_CNN_Tamanho_Filtro')
    lCNNTamanhoFiltro.AdicionarEmbedding(lTamanhoVocabulario, len(lSequencias[0]))
    lCNNTamanhoFiltro.AdicionarConvUnidirecional(128, 'relu', lTamanhoFiltro)
    lCNNTamanhoFiltro.AdicionarPooling()
    lCNNTamanhoFiltro.AdicionarDense(len(set(lLinguas)), 'softmax')
    lCNNTamanhoFiltro.Compilar('sparse_categorical_crossentropy', 'adam', ['accuracy'])

    # Treinando o modelo
    lHistoricoFiltroTamanhoCNN = lCNNTamanhoFiltro.Treinamento(lSequenciasTreino, lLinguasTreinoCodificado, 
                                                              (lSequenciasTeste, lLinguasTesteCodificado))

    lCNNTamanhoFiltro.Salvar(f'modelCNN_filter_size_{lTamanhoFiltro}')

    # Avaliação do modelo
    lPredicaoY = lCNNTamanhoFiltro.Modelo.predict(lSequenciasTeste)
    lPredicaoClassesY = np.argmax(lPredicaoY, axis = 1)
    lAcuracia = accuracy_score(lLinguasTesteCodificado, lPredicaoClassesY)

    print(f"Acurácia do modelo CNN (filtro de tamanho {lTamanhoFiltro}): {lAcuracia}")
    Helper.MostrarPerdaPlot(lHistoricoFiltroTamanhoCNN, 'CNN Filter Size Model')

    lClassesUnique = np.unique(lLinguas)
    lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoClassesY)
    Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'CNN Filter Size Model')
    

# Modelo com tamanho do passo diferente
lFiltroPassosLista = [1, 2, 3]

for lFiltroPasso in lFiltroPassosLista:
    # Criando um modelo CNN com a variação do tamanho do passo
    lCNNTamanhoPasso = Modelo('Modelo_CNN_Tamanho_Passo')
    lCNNTamanhoPasso.AdicionarEmbedding(lTamanhoVocabulario, len(lSequencias[0]))
    lCNNTamanhoPasso.AdicionarConvUnidirecional(128, 'relu', 5, lFiltroPasso)
    lCNNTamanhoPasso.AdicionarPooling()
    lCNNTamanhoPasso.AdicionarDense(len(set(lLinguas)), 'softmax')
    lCNNTamanhoPasso.Compilar('sparse_categorical_crossentropy', 'adam', ['accuracy'])

    # Treinando o modelo
    lHistoricoFiltroCNNPasso = lCNNTamanhoPasso.Treinamento(lSequenciasTreino, lLinguasTreinoCodificado,
                                                                  (lSequenciasTeste, lLinguasTesteCodificado))    

    lCNNTamanhoPasso.Salvar(f'modelCNN_filter_stride_{lFiltroPasso}')

    # Avaliação do modelo
    lPredicaoY = lCNNTamanhoPasso.Modelo.predict(lSequenciasTeste)
    lPredicaoClassesY = np.argmax(lPredicaoY, axis = 1)
    lAcuracia = accuracy_score(lLinguasTesteCodificado, lPredicaoClassesY)

    print(f"Acurácia do modelo CNN (passo de tamanho {lFiltroPasso}): {lAcuracia}")
    Helper.MostrarPerdaPlot(lHistoricoFiltroCNNPasso , 'CNN Filter Stride Model')

    lClassesUnique = np.unique(lLinguas)
    lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoClassesY)
    Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'CNN Filter Stride Model')

# Modelo DNNs (Rede Neural Profunda) com número de camadas ocultas

lDNNLayers = Modelo('Modelo_DNN_Layers')
lDNNLayers.AdicionarInput((len(lSequencias[0]), ))
lDNNLayers.AdicionarDense()
lDNNLayers.AdicionarDense()
lDNNLayers.AdicionarDense()
lDNNLayers.AdicionarDense(len(set(lLinguas)), 'softmax')
lDNNLayers.Compilar('sparse_categorical_crossentropy', 'adam', ['accuracy'])

# Treinando modelo
lHistoricoDNNOCamadaOculta = lDNNLayers.Treinamento(lSequenciasTreino, lLinguasTreinoCodificado,
                                                   (lSequenciasTeste, lLinguasTesteCodificado)) 

lPredicaoLayersY = lDNNLayers.Modelo.predict(lSequenciasTeste)
lPredicaoLayersYClasses = np.argmax(lPredicaoLayersY, axis = 1)
lAcuraciaLayers = accuracy_score(lLinguasTesteCodificado, lPredicaoLayersYClasses)

print(f"Acurácia do modelo DNN Ocult Model: {lAcuraciaLayers}")
Helper.MostrarPerdaPlot(lHistoricoDNNOCamadaOculta, 'DNN Ocult Model')

lClassesUnique = np.unique(lLinguas)
lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoLayersYClasses)
Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'DNN Ocult Model')
lDNNLayers.Salvar('modelDNN_layers')


# Modelo DNNs (Rede Neural Profunda) com diferentes funções de ativação nas camadas ocultas
lDNNCamadasOcultas = Modelo('Modelo_CNN_Camadas_Ocultas')
lDNNCamadasOcultas.AdicionarInput((len(lSequencias[0]),))
lDNNCamadasOcultas.AdicionarDense()
lDNNCamadasOcultas.AdicionarDense(pAtivacao = 'tanh')
lDNNCamadasOcultas.AdicionarDense(pAtivacao = 'sigmoid')
lDNNCamadasOcultas.AdicionarDense(len(set(lLinguas)), 'softmax')
lDNNCamadasOcultas.Compilar('sparse_categorical_crossentropy', 'adam', ['accuracy'])

# Treinando modelo
lHistoricoDNNCamadaOculta = lDNNCamadasOcultas.Treinamento(lSequenciasTreino, lLinguasTreinoCodificado, 
                                                          (lSequenciasTeste, lLinguasTesteCodificado))

lDNNPredicaoY = lDNNCamadasOcultas.Modelo.predict(lSequenciasTeste)
lPredicaoDNNClassesY = np.argmax(lDNNPredicaoY, axis = 1)
lAcuraciaDNN = accuracy_score(lLinguasTesteCodificado, lPredicaoDNNClassesY)

print(f"Acurácia do modelo DNN Ocult Activation: {lAcuraciaDNN}")
Helper.MostrarPerdaPlot(lHistoricoDNNCamadaOculta, 'DNN Ocult Ativation Model')

lClassesUnique = np.unique(lLinguas)
lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoDNNClassesY)
Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'DNN Ocult Ativation Model')

lDNNCamadasOcultas.Salvar('modelDNN_ocult_activation')


# Modelo DNNs (Rede Neural Profunda) com variação da taxa de aprendizado e função de ativação
lDNNTaxaAprendizado = Modelo('Modelo_DNN_Taxa_Aprendizado')
lDNNTaxaAprendizado.AdicionarInput((len(lSequencias[0]),))
lDNNTaxaAprendizado.AdicionarDense()
lDNNTaxaAprendizado.AdicionarDense()
lDNNTaxaAprendizado.AdicionarDense(len(set(lLinguas)), 'softmax')

# Configuração personalizada do otimizador com taxa de aprendizado
lOtimizador = lDNNTaxaAprendizado.CustomizarOtimizador(0.001)
lDNNTaxaAprendizado.Compilar('sparse_categorical_crossentropy', lOtimizador, ['accuracy'])

# Treinando modelo
lHistoricoDNNAprendizado = lDNNTaxaAprendizado.Treinamento(lSequenciasTreino, lLinguasTreinoCodificado,
                                                          (lSequenciasTeste, lLinguasTesteCodificado))

lPredicaoY = lDNNTaxaAprendizado.Modelo.predict(lSequenciasTeste)
lPredicaoClassesY = np.argmax(lPredicaoY, axis = 1)
lAcuracia = accuracy_score(lLinguasTesteCodificado, lPredicaoClassesY)
print(f"Acurácia do modelo DNN Rate Learn: {lAcuracia}")
Helper.MostrarPerdaPlot(lHistoricoDNNAprendizado, 'DNN Rate Learn Model')

lClassesUnique = np.unique(lLinguas)
lMatrizConfusao = confusion_matrix(lLinguasTesteCodificado, lPredicaoClassesY)
Helper.MostrarMatrixConfusao(lMatrizConfusao, lClassesUnique, 'DNN Rate Learn Model')

lDNNTaxaAprendizado.Salvar('model_DNN_Taxa_Aprendizado')


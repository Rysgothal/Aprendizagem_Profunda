import os
import pandas as pd
from sklearn.model_selection import train_test_split
from helper.ap_helper import Helper

def SalvarArquivoCSV(pDataset, pPasta, pDados):
    # Salvando os arquivos CSV de treino e teste
    for lClasseDataset in pDataset['Classe'].unique():
        Helper.CriarPasta(os.path.join(pPasta, lClasseDataset))

        lTreinoDadosClasse = pDados[pDados['Classe'] == lClasseDataset]
        lArquivoTreino = os.path.join(pPasta, lClasseDataset, 'treino.csv')  # Caminhos para os arquivos de treino e teste
        lTreinoDadosClasse.to_csv(lArquivoTreino, index = False, mode = 'w') # Substitui os arquivos se eles já existirem

def SepararPastasDataset(pTexto: str):
    lCaminho = 'Textos'    # Caminho da pasta onde os conjuntos serão salvos
    lPorcentagem = 0.5     # Proporção do Dataset a ser usado como teste
    lRandomState = 42      # Geração de números aleatórios

    lDiretorio = os.path.dirname(os.path.realpath(__file__))
    lLinhas = pTexto.split('\n')
    lDados = []

    # Para cada linha, adiciona à lista de dados
    for lLinha in lLinhas:
        if not lLinha.strip():
            continue 

        lPartes = lLinha.split(';')

        if len(lPartes) >= 2:
            lTexto = lPartes[0].strip()
            lClasse = lPartes[1].strip()
            lDados.append({'Texto': lTexto, 'Classe': lClasse})
        else:
            lDados.append({'Texto': lLinha, 'Classe': 'SemClasse'})

    lDataset = pd.DataFrame(lDados)                                                            # Converte a lista de dados em um DataFrame
    lDataset = lDataset.sample(frac = 1, random_state = lRandomState).reset_index(drop = True) # Embaralha o dataset antes de dividir  
    
    lPastaTreino = Helper.CriarPasta(os.path.join(lDiretorio, lCaminho, 'train'))
    lPastaTeste = Helper.CriarPasta(os.path.join(lDiretorio, lCaminho, 'test'))

    # Divide o dataset em conjuntos de treino e teste
    lTreinoDados, lTesteDados = train_test_split(lDataset, test_size = lPorcentagem, random_state = lRandomState)

    SalvarArquivoCSV(lDataset, lPastaTreino, lTreinoDados)
    SalvarArquivoCSV(lDataset, lPastaTeste, lTesteDados) 

lCaminhoDataset = os.path.abspath(os.path.dirname(__file__)) + '/dataset.txt'
with open(lCaminhoDataset, encoding = 'utf-8') as lArquivo:
    SepararPastasDataset(lArquivo.read())
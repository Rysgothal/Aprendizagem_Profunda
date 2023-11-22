import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ap_helper import Helper

def ConfigurarDataSet(pTexto):
    lLinhas = pTexto.split('\n') 
    lDados = []
    
    for lLinha in lLinhas:
        if not lLinha.strip():
            continue
        
        lPartes = lLinha.split(';')
        
        if len(lPartes) >= 2:
            pTexto = lPartes[0].strip()
            lClasse = lPartes[1].strip()
            lDados.append({'Texto': pTexto, 'Classe': lClasse})
        else:
            lDados.append({'Texto': lLinha, 'Classe': 'SemClasse'})
    
    return lDados

def ConfigurarPastasDataSet(pDados: list):
    lDataSet = pd.DataFrame(pDados)
    lDataSet = lDataSet.sample(frac = 1, random_state = 42).reset_index(drop = True)
    
    lPastaTreino = Helper.DiretorioAtual() + '/helper/Textos/train'
    lPastaTeste = Helper.DiretorioAtual() + '/helper/Textos/test'
    Helper.CriarPasta(lPastaTreino)
    Helper.CriarPasta(lPastaTeste)
    
    lTreinoDados, lTesteDados = train_test_split(lDataSet, test_size = 0.5, random_state = 42)
    
    for lClasse in lDataSet['Classe'].unique():
        Helper.CriarPasta(os.path.join(lPastaTreino, lClasse))
        Helper.CriarPasta(os.path.join(lPastaTeste, lClasse))
    
    for lClasse in lDataSet['Classe'].unique():
        lTreinoDadosClasse = lTreinoDados[lTreinoDados['Classe'] == lClasse]
        lTesteDadosClasse = lTesteDados[lTesteDados['Classe'] == lClasse]
      
        lArquivoTreino = os.path.join(lPastaTreino, lClasse, 'treino.csv')
        lArquivoTeste = os.path.join(lPastaTeste, lClasse, 'teste.csv')
        lTreinoDadosClasse.to_csv(lArquivoTreino, index = False, mode = 'w')
        lTesteDadosClasse.to_csv(lArquivoTeste, index = False, mode = 'w')

def EmbaralharDataSet(pTexto: str):
    lDados = ConfigurarDataSet(pTexto)
    ConfigurarPastasDataSet(lDados)
    
def Embaralhar():    
    with open("dataset.txt", "r", encoding="utf-8") as file:
        EmbaralharDataSet(file.read())
        
        
Embaralhar()      
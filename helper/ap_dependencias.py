import subprocess
import sys

FErros = []

# Função para Instalar Bibliotecas [ Genérico ]
def Instalar(pPacote):
    try:
        subprocess.run(['pip', 'install', pPacote,'--user'], check = True)
    except Exception as e:
        FErros.append([pPacote, e])
    
# =========================== #
#   instalando dependencias   #
# =========================== #

def VerificandoDependencias():
    print('Verificando pacotes...\n')
    
    try:
        import pandas
    except ModuleNotFoundError:
        Instalar('pandas')    
        
    try:
        import sklearn 
    except ModuleNotFoundError:
        Instalar('scikit-learn') 

    try:
        import tensorflow
    except ModuleNotFoundError:
        Instalar('tensorflow')        
        
    try:
        import numpy
    except ModuleNotFoundError:
        Instalar('numpy')
        
    try:
        import seaborn
    except ModuleNotFoundError:
        Instalar('seaborn') 

VerificandoDependencias()

if FErros:
    print('Erros:')
    for lErro in FErros:
        print(f'Pacote: {lErro[0]}, Erro: {lErro[1]}')

    sys.exit()


import subprocess

FErros = []

# Função para Instalar Bibliotecas [ Genérico ]
def Instalar(pPacote):
    try:
        subprocess.run(['pip', 'install', pPacote,'--user'], check=True)
    except Exception as e:
        FErros.append([pPacote, e])
    
# =========================== #
#   instalando dependencias   #
# =========================== #

def VerificandoDependencias():
    print('Verificando pacotes...\n')
    
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
        import pandas
    except ModuleNotFoundError:
        Instalar('pandas')    
    try:
        import seaborn
    except ModuleNotFoundError:
        Instalar('seaborn') 
VerificandoDependencias()

print("Erros:")
for erro in FErros:
    print(f"Pacote: {erro[0]}, Erro: {erro[1]}")

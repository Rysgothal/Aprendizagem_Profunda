import helper.ap_instalar_curl
import helper.ap_instalar_pip
from helper.ap_dependencias import FErros

def MensagemParaFalhas():
    print('Foi encontrado alguma inconformidade na instalação das dependencias, verifique...')
    print(f'\nInformações: ')
    for lPacote, lMensagem in FErros:
        print(f'O pacote "{lPacote}": \n{lMensagem}\n\n')
    print('Sugestões: ')
    print(' Verifique se o "PIP" está atualizado com: sudo apt install python3-pip')
    print(' Pacotes para funcionamento do programa:')
    print(' \t -> $ pip install scikit-learn')
    print(' \t -> $ pip install tensorflow')
    print(' \t -> $ pip install seaborn')
    print(' \t -> $ pip install pandas')
    print(' \t -> $ pip install numpy\n')
    print(' \n Saindo da aplicação...')

if FErros:
    MensagemParaFalhas()
    exit()

print('\nTodas as depencias estão OK.')

try:
    print('\nAjustando pastas...')
    import helper.ap_embaralhar
    print('Pastas ajustadas para prosseguir')
    print('Iniciando treinamento...')       
    import helper.ap_treinamento
    print('\nTreinamento concluido')  
except Exception as e:
    print('Houve uma inconsistencia ao executar o programa...')
    print(f'Verifique:\n {e}')

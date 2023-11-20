# Arquivo principal para a execução
from dependencia import FErros

def MensagemParaFalhas():
    print('Foi encontrado alguma inconformidade na instalação das dependencias, verifique...')
    print(f'\nInformações: ')
    for lPacote, lMensagem in FErros:
        print(f'O pacote "{lPacote}": \n{lMensagem}\n\n')
    print('Verifique se o "PIP" está atualizado: sudo apt install python3-pip')
    print('Pacotes para funcionamento:')
    print('\t -> ($ pip install) scikit-learn')
    print('\t -> ($ pip install) tensorflow')
    # print('\t -> python3 -m pip install -U spacy==2.3.8')
    # print('\t -> python3 -m spacy download pt_core_news_sm')
    print('\t -> ($ pip install) seaborn')
    print('\t -> ($ pip install) pandas')
    print('\t -> ($ pip install) numpy')
    # print('\t -> ($ pip install) matplotlib==3.6.2')
    print('\n Saindo da aplicação...')

if FErros:
    MensagemParaFalhas()
    exit()
 
print('\nTodas as depencias estão OK.')
# try:
#     print('Iniciando treinamento...')       
#     import as_treinamento
#     print('Iniciando teste..')       
#     import as_teste
#     print('\nIniciando Bot...')   
#     import as_bot_telegram 
# except Exception as e:
#     print('Houve uma inconsistencia ao executar o programa...')
#     print(f'Verifique:\n {e}')
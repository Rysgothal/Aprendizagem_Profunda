import subprocess
import sys
import os

def InstalarConda():
    import platform
    import urllib.request

    try:
        lSistemaOperacional = platform.system().lower()

        # URL instalador 16 de novembro de 2023
        if lSistemaOperacional != 'linux':
            print('Seu sistema operacional não é compatível para o programa')
            print(f'Seu sistema: {lSistemaOperacional}')
            print('Sistema esperado: Linux')
            sys.exit()

        lURL = 'https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh'
        lNomeInstalador = 'anaconda.sh'
        urllib.request.urlretrieve(lURL, lNomeInstalador)        # Baixar o instalador
        os.chmod(lNomeInstalador, 0o755)                         # Permissões de execução ao instalador
        subprocess.run(['./' + lNomeInstalador], check = True)
        os.remove(lNomeInstalador)
        
        subprocess.run(['clear'])
        print('"Conda" foi instalado com sucesso.')
        print('Rode novamente o programa em outro terminal...')
        sys.exit()
    except Exception as e:
        print(f'Falha ao instalar o gerenciador de pacotes "Conda", tente novamente.\n {e}')

def IniciarAmbienteConda():
    print('Criando ambiente conda...')

    lJaExiste = True
    lAtivarExistente = ''
    lListaAmbientes = subprocess.run(['conda', 'env', 'list'], capture_output = True, text = True)
    lNomeAmbiente = ''

    while lJaExiste:
        lNomeAmbiente = input('Qual será o nome do seu ambiente? Caso vazio será ap_Deep_Learning.\n')
        
        if lNomeAmbiente.strip() == '':
            lNomeAmbiente = 'ap_Deep_Learning'

        if lNomeAmbiente in lListaAmbientes.stdout:
            lAtivarExistente = input(f'O ambiente Conda "{lNomeAmbiente}" já existe. Deseja ativá-lo? (S/N)\n').upper()
            
            if lAtivarExistente == 'S':
                lJaExiste = False
            elif lAtivarExistente == 'N':
                lJaExiste = True
            else: 
                print('Opção inválida, tente novamente.')
                lJaExiste = True
        else:
            lJaExiste = False

    # Crie o ambiente Conda
    if lAtivarExistente != 'S': 
        lAmbiente = subprocess.run(['conda', 'create', '-n', lNomeAmbiente, 'python=3.11.5', '-y'])
        
        if lAmbiente.returncode == 0:
            print(f'\nO ambiente Conda "{lNomeAmbiente}" foi criado com Python 3.11.5')
        else:
            print(f'Falha ao criar o ambiente Conda "{lNomeAmbiente}".')
            print(f'Rode novamente o programa...')
            sys.exit()

    print(f'\nAmbiente pronto para uso, siga as instruções: ')
    print(f' 1 - Ative o ambiente: \t\t\t$ conda activate {lNomeAmbiente}') 
    print(' 2 - Rode o arquivo no ambiente: \t$ python3 ap_rodar.py \n')

def VerificarConda():
    try:
        print('\nVerificando gerenciador de pacotes "Conda"...')
        lArgumentos = ['conda', '--version']
        subprocess.run(lArgumentos)
        print('O gerenciador de pacotes "Conda" está instalado.\n')

    except FileNotFoundError:
        print('O gerenciador de pacotes "Conda" não foi encontrado, prosseguindo com a instalação, por favor aguarde.')
        print('Isso pode demorar algum tempo...')
        InstalarConda()

    except Exception as e:
        print(f'Falha ao verificar o gerenciador de pacotes "Conda", tente novamente.\n {e}')
        sys.exit()


VerificarConda()
IniciarAmbienteConda()
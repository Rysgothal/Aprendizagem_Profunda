# Verifica se o pacote Curl está presente no ambiente
import subprocess

def InstalarCurl():
    try:
        print('Instalando pacote "Curl" ...')
        lArgumentos = ['sudo', 'apt', 'update']
        subprocess.run(lArgumentos, check = True)

        lArgumentos = ['sudo', 'apt','install', 'curl']
        subprocess.run(lArgumentos, check = True)
        
        print('O pacote "Curl" foi instalado com sucesso.')
    except Exception as e:
        print(f'Houve uma falha ao instalar o pacote "curl": \n{e}')
        print('\nInstalação manual requerida:')
        print(' sudo apt update')
        print(' sudo apt install curl')
        print(' curl --version')

def VerificarCurl():
    try:
        print('Verificando pacote "Curl" ...')
        lArgumentos = ['curl', '--version']
        subprocess.run(lArgumentos, stdout = subprocess.PIPE, stderr = subprocess.PIPE, check = True)

        print('O "Curl" está instalado.')
    except FileNotFoundError:
        print('"Curl" não foi encontrado.')
        InstalarCurl()

VerificarCurl()

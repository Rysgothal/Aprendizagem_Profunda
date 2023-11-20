import subprocess
import sys

def PipInstalado():
    try:
        import pip
        return True
    except ImportError:
        return False

def InstalarPip():
    try:
        subprocess.check_call(['conda', 'install', 'pip'])
        print('pip foi instalado com sucesso.')
    except Exception as e:
        print(f'Erro ao instalar pip: {e}')
        print('Verifique, e rode novamente o programa')
        sys.exit()

if not PipInstalado():
    print("pip não está instalado. Instalando...")
    InstalarPip()
else:
    print("pip já está instalado.")
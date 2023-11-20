import subprocess

def is_pip_installed():
    try:
        import pip
        return True
    except ImportError:
        return False

def install_pip():
    try:
        # Use o subprocess para baixar e instalar o pip
        # subprocess.check_call(["curl", "https://bootstrap.pypa.io/get-pip.py", "-o", "get-pip.py"])
        subprocess.check_call(["conda", "install","pip"])
        print("pip foi instalado com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao instalar pip: {e}")
        exit(1)

if not is_pip_installed():
    print("pip não está instalado. Instalando...")
    install_pip()
else:
    print("pip já está instalado.")

# Agora você pode usar o pip para instalar pacotes, se necessário.

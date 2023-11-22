import subprocess

def check_and_install_curl():
    try:
        # Tente executar o comando 'curl --version' para verificar se o curl está instalado.
        subprocess.run(['curl', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("curl já está instalado.")
    except FileNotFoundError:
        # Se o comando 'curl' não for encontrado, o curl não está instalado. Vamos instalá-lo.
        try:
            subprocess.run(['sudo', 'apt', 'update'], check=True)
            subprocess.run(['sudo', 'apt', 'install', 'curl'], check=True)
            print("curl foi instalado com sucesso.")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao instalar curl: {e}")

if __name__ == "__main__":
    check_and_install_curl()

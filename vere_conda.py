import subprocess

# Nome do ambiente Conda que você deseja criar
env_name = "deep_learn"

# Verifique se o ambiente Conda já existe
check_env_exists = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
if env_name in check_env_exists.stdout:
    print(f"O ambiente Conda '{env_name}' já existe.")
else:
    # Crie o ambiente Conda
    create_env = subprocess.run(["conda", "create", "-n", env_name, "python=3.11.5", "-y"])
    if create_env.returncode == 0:
        print(f"O ambiente Conda '{env_name}' foi criado com Python 3.11.5")
    else:
        print(f"Falha ao criar o ambiente Conda '{env_name}'.")
        exit(1)

# Ative o ambiente Conda
activate_env = subprocess.run(["conda", "activate", env_name], shell=True)
if activate_env.returncode == 0:
    print(f"Ambiente Conda '{env_name}' ativado.")
else:
    print(f"Falha ao ativar o ambiente Conda '{env_name}'.")
    exit(1)

# Agora você está dentro do ambiente Conda com Python ativado
# Você pode executar seu código Python ou outras tarefas aqui

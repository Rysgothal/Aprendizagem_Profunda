import os
import pandas as pd
from sklearn.model_selection import train_test_split
from io import StringIO

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import random  # Adiciona o módulo random

def dividir_e_salvar_dataset(texto, pasta_base='dataset', test_size=0.5, random_state=42):
    """
    Divide e embaralha o dataset em conjuntos de treino e teste, salvando os resultados em pastas.

    Parâmetros:
    - texto: String contendo o dataset.
    - pasta_base: Caminho da pasta base onde os conjuntos serão salvos.
    - test_size: Proporção do dataset a ser usado como conjunto de teste.
    - random_state: Semente para a geração de números aleatórios.
    """

    # Obtém o caminho do diretório do script
    diretorio_script = os.path.dirname(os.path.realpath(__file__))

    # Divide o texto em linhas
    linhas = texto.split('\n')

    # Lista para armazenar os dados
    dados = []

    # Para cada linha, adiciona à lista de dados
    for linha in linhas:
        if linha.strip():  # Ignora linhas em branco
            # Divide a linha pelo ponto e vírgula
            partes = linha.split(';')

            # Verifica se há pelo menos dois elementos após a divisão
            if len(partes) >= 2:
                texto = partes[0].strip()
                classe = partes[1].strip()
                dados.append({'Texto': texto, 'Classe': classe})
            else:
                # Se não houver classe, adiciona à lista com a classe padrão
                dados.append({'Texto': linha, 'Classe': 'SemClasse'})

    # Converte a lista de dados em um DataFrame
    dataset = pd.DataFrame(dados)

    # Embaralha o dataset antes de dividir
    dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Cria as pastas de treino e teste dentro do diretório do script
    pasta_treino = os.path.join(diretorio_script, pasta_base, 'train')
    pasta_teste = os.path.join(diretorio_script, pasta_base, 'test')

    os.makedirs(pasta_treino, exist_ok=True)
    os.makedirs(pasta_teste, exist_ok=True)

    # Divide o dataset em conjuntos de treino e teste
    train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=random_state)

    # Cria subpastas para cada classe dentro das pastas train e test
    for classe in dataset['Classe'].unique():
        os.makedirs(os.path.join(pasta_treino, classe), exist_ok=True)
        os.makedirs(os.path.join(pasta_teste, classe), exist_ok=True)

    # Salva os arquivos CSV de treino e teste
    for classe in dataset['Classe'].unique():
        train_data_classe = train_data[train_data['Classe'] == classe]
        test_data_classe = test_data[test_data['Classe'] == classe]

        # Caminhos para os arquivos de treino e teste
        arquivo_treino = os.path.join(pasta_treino, classe, 'treino.csv')
        arquivo_teste = os.path.join(pasta_teste, classe, 'teste.csv')

        # Substitui os arquivos se eles já existirem
        train_data_classe.to_csv(arquivo_treino, index=False, mode='w')
        test_data_classe.to_csv(arquivo_teste, index=False, mode='w')
    print("Embaralhar!")

# Lê o conteúdo do arquivo
with open("dataset.txt", "r", encoding="utf-8") as file:
    texto_do_arquivo = file.read()
# Chama a função para dividir e salvar o dataset
dividir_e_salvar_dataset(texto_do_arquivo)
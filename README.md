# Machine Learning Aplicado à Saúde

# Descrição do Projeto
Aplicação web interativa demonstrando técnicas de Machine Learning aplicadas a dados de saúde cardíaca. O projeto utiliza um dataset médico real para prever doenças cardíacas usando aprendizado supervisionado e não-supervisionado.

# Funcionalidades

Análise Exploratória: Visualização e estatísticas dos dados médicos

ML Supervisionado: Classificação para prever doenças cardíacas (Random Forest)

ML Não-Supervisionado: Clusterização com K-Means para descobrir padrões

Sobre os Dados: Informações detalhadas do dataset

# Tecnologias Utilizadas

Python 3.8+

Streamlit

Scikit-learn

Pandas, NumPy

Matplotlib, Seaborn

# Dataset

Medicaldataset.csv - Dados de pacientes com indicadores de saúde cardíaca:

Variáveis: Age, Gender, Heart rate, Blood pressure, Blood sugar, CK-MB, Troponin

Target: Result (positive/negative para doença cardíaca)

# Como Executar

Pré-requisitos
Python 3.8 ou superior

Pip (gerenciador de pacotes do Python)

# Instalação
Clone o repositório:
git clone <seu-repositorio>
cd ML-Saude-doencas-cardiacas

Instale as dependências:
pip install streamlit pandas numpy matplotlib seaborn scikit-learn

Execute a aplicação:
python -m streamlit run app.py

# Nota Importante

No Windows, use sempre:
python -m streamlit run app.py
Execução Alternativa (Ambiente Virtual - Recomendado)

# Criar ambiente virtual

python -m venv venv

# Ativar no Windows

venv\Scripts\activate

# Instalar dependências

pip install -r requirements.txt

# Executar

python -m streamlit run app.py

Estrutura do Projeto
app.py                 
Medicaldataset.csv    
requirements.txt       
README.md             

# Como Usar

Sobre os Dados": Visualize informações do dataset

Análise Exploratória": Explore estatísticas e visualizações

ML Supervisionado": Treine modelos de classificação

ML Não-Supervisionado": Execute clusterização

# Resultados

Acurácia de classificação acima de 80%

Identificação de features mais importantes para diagnóstico

Clusters com padrões médicos significativos

# Deploy

Para deploy no Streamlit Cloud:

Suba o projeto para GitHub

Conecte no Streamlit Cloud

Configure para executar app.py

# Autor

Bruno Alexandre - Projeto de Machine Learning Aplicado à Saúde

# Licença

Este projeto é para fins educacionais.

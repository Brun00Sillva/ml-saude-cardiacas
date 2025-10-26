import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML em Saúde", page_icon=":heart:", layout="wide")

st.title("Machine Learning Aplicado à Saúde")
st.subheader("Análise de Doenças Cardíacas ")

# Sidebar para navegação #
page = st.sidebar.selectbox("Navegação", 
[" Sobre os Dados", " Análise Exploratória", " ML Supervisionado", " ML Não-Supervisionado"])

# Carregar dados do arquivo local #
@st.cache_data
def load_data():
    try:
        # Carrega do arquivo local #
        df = pd.read_csv('Medicaldataset.csv')
        
        # Verifica e limpa os dados #
        st.sidebar.info(f" Dataset: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
    except Exception as e:
        st.error(f" Erro ao carregar arquivo: {e}")
        # Dataset de fallback vazio #
        df = pd.DataFrame()
    
    return df

df = load_data()

if df.empty:
    st.error("Não foi possível carregar os dados. Verifique se o arquivo 'Dataset.csv' está na pasta correta.")
    st.stop()

if page == " Sobre os Dados":
    st.header("Informações sobre o Dataset Médico")
    
    st.write("# Primeiras 10 linhas:")
    st.dataframe(df.head(10))
    
    st.write("# Informações gerais:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Formato do dataset:**")
        st.write(f"Linhas: {df.shape[0]}")
        st.write(f"Colunas: {df.shape[1]}")
        
        st.write("**Colunas disponíveis:**")
        for col in df.columns:
            st.write(f"- {col}")
    
    with col2:
        st.write("**Tipos de dados:**")
        types_df = pd.DataFrame({
        'Coluna': df.columns.tolist(),
        'Tipo de Dado': [str(dtype) for dtype in df.dtypes]
    })
        st.dataframe(types_df, use_container_width=True, hide_index=True)
        
        st.write("**Valores faltantes:**")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            st.dataframe(missing[missing > 0])
        else:
            st.success(" Nenhum valor faltante encontrado!")
    
    with col3:
        st.write("**Estatísticas da variável target:**")
        if 'Result' in df.columns:
            result_counts = df['Result'].value_counts()
            st.write(result_counts)
            
            fig, ax = plt.subplots()
            result_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_title("Distribuição de Resultados")
            st.pyplot(fig)

elif page == " Análise Exploratória":
    st.header(" Análise Exploratória dos Dados")
    
    # Verifica se a coluna Result existe #
    if 'Result' not in df.columns:
        st.error("Coluna 'Result' não encontrada no dataset!")
        st.stop()
    
    st.write("# Distribuição da Variável Target")
    col1, col2 = st.columns(2)
    
    with col1:
        result_counts = df['Result'].value_counts()
        st.dataframe(result_counts)
        
    with col2:
        fig, ax = plt.subplots()
        result_counts.plot(kind='bar', ax=ax, color=['red', 'green'])
        ax.set_title("Distribuição de Resultados")
        ax.set_xlabel("Resultado")
        ax.set_ylabel("Quantidade")
        plt.xticks(rotation=0)
        st.pyplot(fig)
    
    st.write("# Estatísticas Descritivas")
    st.dataframe(df.describe())
    
    # Análise por resultado #
    st.write("# Estatísticas por Resultado")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats_by_result = df.groupby('Result')[numeric_cols].mean()
    st.dataframe(stats_by_result)
    
    # Matriz de correlação #
    st.write("# Matriz de Correlação")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title("Matriz de Correlação")
    st.pyplot(fig)
    
    # Histogramas interativos #
    st.write("# Distribuição das Variáveis Numéricas")
    selected_col = st.selectbox("Selecione uma coluna para ver a distribuição:", numeric_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        df[selected_col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_title(f"Distribuição de {selected_col}")
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Frequência")
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        df.boxplot(column=selected_col, by='Result', ax=ax)
        ax.set_title(f"{selected_col} por Resultado")
        plt.suptitle('') 
        st.pyplot(fig)

elif page == " ML Supervisionado":
    st.header(" Aprendizado Supervisionado - Previsão de Doença Cardíaca")
    
    if 'Result' not in df.columns:
        st.error("Coluna 'Result' não encontrada!")
        st.stop()
    
    # Preparar dados #
    X = df.drop('Result', axis=1)
    y = df['Result']
    
    # Converter variável target para numérica se necessário #
    if y.dtype == 'object':
        y = y.map({'negative': 0, 'positive': 1})
        st.info("Variável target convertida: negative→0, positive→1")
    
    st.write("# Configuração do Modelo")
    
    # Seleção de features #
    available_features = X.columns.tolist()
    features = st.multiselect("Selecione as features para o modelo:", 
                             available_features, 
                             default=available_features)
    
    if not features:
        st.warning("Selecione pelo menos uma feature!")
        st.stop()
    
    X_selected = X[features]
    
    # Verificar e tratar dados não numéricos #
    if X_selected.select_dtypes(include=['object']).shape[1] > 0:
        st.warning("⚠️ Convertendo variáveis categóricas para numéricas...")
        X_selected = pd.get_dummies(X_selected, drop_first=True)
    
    # Split dos dados #
    test_size = st.slider("Tamanho do conjunto de teste:", 0.1, 0.5, 0.3)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Configuração do modelo #
    n_estimators = st.slider("Número de árvores (n_estimators):", 50, 200, 100)
    
    # Modelo #
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=42,
        max_depth=10
    )
    
    with st.spinner("Treinando o modelo..."):
        model.fit(X_train, y_train)
        
        # Previsões #
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
    # Métricas #
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Acurácia", f"{accuracy:.2%}")
    with col2:
        st.metric("Amostras de Treino", len(X_train))
    with col3:
        st.metric("Amostras de Teste", len(X_test))
    with col4:
        st.metric("Features Utilizadas", len(features))
    
    # Matriz de confusão #
    st.write("# Matriz de Confusão")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negativo', 'Positivo'],
                yticklabels=['Negativo', 'Positivo'])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão")
    st.pyplot(fig)
    
    # Relatório de classificação #
    st.write("# Relatório de Classificação")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))
    
    # Importância das features #
    if hasattr(model, 'feature_importances_'):
        st.write("# Importância das Features")
        
        # Para features one-hot encoded, agrupar por feature original #
        feature_importance_dict = {}
        for feature, importance in zip(X_selected.columns, model.feature_importances_):
            # Extrair nome base para features one-hot #
            base_feature = feature.split('_')[0] if '_' in feature else feature
            if base_feature in feature_importance_dict:
                feature_importance_dict[base_feature] += importance
            else:
                feature_importance_dict[base_feature] = importance
        
        feature_importance = pd.DataFrame({
            'feature': list(feature_importance_dict.keys()),
            'importance': list(feature_importance_dict.values())
        }).sort_values('importance', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Top 10 Features Mais Importantes:")
            st.dataframe(feature_importance.head(10))
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=ax)
            ax.set_title("Top 10 Features Mais Importantes")
            st.pyplot(fig)

else:  # ML Não-Supervisionado #
    st.header(" Aprendizado Não-Supervisionado")
    
    st.write("### Clusterização com K-Means")
    
    # Seleção de variáveis numéricas #
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("É necessário ter pelo menos 2 colunas numéricas para clusterização")
        st.stop()
    
    selected_cols = st.multiselect("Selecione variáveis para clusterização:", 
                                  numeric_cols, 
                                  default=numeric_cols[:3])
    
    if len(selected_cols) < 2:
        st.warning("Selecione pelo menos 2 variáveis para clusterização")
        st.stop()
    
    X_cluster = df[selected_cols].dropna()
    
    if len(X_cluster) == 0:
        st.error("Não há dados suficientes após remover valores faltantes")
        st.stop()
    
    # Normalização #
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # K-Means #
    k = st.slider("Número de clusters (K):", 2, 6, 3)
    
    with st.spinner("Executando clusterização..."):
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        df_cluster = df.loc[X_cluster.index].copy()
        df_cluster['cluster'] = clusters
    
    # Visualização #
    st.write("# Visualização dos Clusters")
    
    if len(selected_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.7)
        ax.set_xlabel(selected_cols[0])
        ax.set_ylabel(selected_cols[1])
        ax.set_title(f"Clusters K-Means (K={k})")
        plt.colorbar(scatter, label='Cluster')
        st.pyplot(fig)
    
    # Estatísticas dos clusters #
    st.write("# Estatísticas dos Clusters")
    cluster_stats = df_cluster.groupby('cluster')[selected_cols].mean()
    st.dataframe(cluster_stats)
    
    # Tamanho dos clusters #
    st.write("# Distribuição dos Clusters")
    cluster_sizes = df_cluster['cluster'].value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        cluster_sizes.plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_title("Quantidade de Amostras por Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Número de Amostras")
        plt.xticks(rotation=0)
        st.pyplot(fig)
    
    with col2:
        if 'Result' in df_cluster.columns:
            # Relação entre clusters e resultado real #
            cross_tab = pd.crosstab(df_cluster['cluster'], df_cluster['Result'])
            fig, ax = plt.subplots()
            cross_tab.plot(kind='bar', ax=ax)
            ax.set_title("Relação Cluster vs Resultado Real")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Quantidade")
            plt.xticks(rotation=0)
            st.pyplot(fig)


st.markdown("**Machine Learning Aplicado à Saúde** - Análise de dados cardíacos")
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Carregar os dados
dados_frame1 = pd.read_csv('https://relacoesinstitucionais.com.br/Fotos/Temp/dados_csv_Boston_sem_outliers.csv')
dados_frame2 = pd.read_csv('https://relacoesinstitucionais.com.br/Fotos/Temp/dados_csv_New_York_sem_outliers.csv')
dados_frame3 = pd.read_csv('https://relacoesinstitucionais.com.br/Fotos/Temp/dados_csv_Philadelphia_sem_outliers.csv')

# Sidebar para filtros
st.sidebar.header('Escolher a região, cluster, loja e item')

# Seleção da região
region_selecionado = st.sidebar.selectbox('Selecione a Região', options=['Boston', 'New_York', 'Philadelphia'])

# Carrega o DataFrame correto com base na região selecionada
if region_selecionado == 'Boston':
    dados_filtrados = dados_frame1
elif region_selecionado == 'New_York':
    dados_filtrados = dados_frame2
elif region_selecionado == 'Philadelphia':
    dados_filtrados = dados_frame3

# Seleção do cluster
cluster_selecionado = st.sidebar.selectbox('Selecione o cluster', options=dados_filtrados['Cluster'].unique())

# Filtra os dados pelo cluster selecionado
dados_filtrados_cluster = dados_filtrados[dados_filtrados['Cluster'] == cluster_selecionado]

# Seleção da loja
store_selecionado = st.sidebar.selectbox('Selecione a Loja', options=dados_filtrados_cluster['store'].unique())

# Filtra os dados pela loja selecionada
dados_filtrados_loja = dados_filtrados_cluster[dados_filtrados_cluster['store'] == store_selecionado]

# Seleção do item
item_selecionado = st.sidebar.selectbox('Selecione o Item', options=dados_filtrados_loja['item'].unique())

# Filtrar os dados
dados_filtrados_cluster_selecionado2 = dados_filtrados[(dados_filtrados['Cluster'] == cluster_selecionado) & (dados_filtrados['item'] == item_selecionado) & (dados_filtrados['store'] == store_selecionado)]

# Agrupar os dados
dados_filtrados_cluster_selecionado3 = dados_filtrados_cluster_selecionado2.groupby(['year_month'])['SOMA'].sum().reset_index()

# Botão para resetar filtros
if st.sidebar.button('Resetar Filtros'):
    dados_filtrados_cluster_selecionado2 = dados_filtrados
    dados_filtrados_cluster_selecionado3 = dados_filtrados_cluster_selecionado2.groupby(['year_month'])['SOMA'].sum().reset_index()

# Preparar os dados para o Prophet
dados_prophet = dados_filtrados_cluster_selecionado3.rename(columns={'year_month': 'ds', 'SOMA': 'y'})

#
df_real = pd.DataFrame(dados_prophet)
df_merged = pd.merge(df_real, forecast[['ds', 'yhat']], on='ds', how='left')
df_merged



# Botão para executar a previsão
if st.button('Executar Previsão'):
    modelo = Prophet()
    modelo.fit(dados_prophet)
    futuro = modelo.make_future_dataframe(periods=20, freq='MS')
    previsao = modelo.predict(futuro)

    # Plotar os resultados
    fig = px.line(previsao, x='ds', y='yhat', title='Previsão com Prophet')
    st.plotly_chart(fig)


    # Plotando os dados
    plt.figure(figsize=(10, 5))
    plt.plot(df_merged['ds'], df_merged['y'], label='Valor Real', marker='o')
    plt.plot(df_merged['ds'], df_merged['yhat'], label='Valor Predito', marker='x')

    # Adicionando título e rótulos
    plt.title('Comparação entre Valor Real e Valor Predito')
    plt.xlabel('Data')
    plt.ylabel('Valores')
    plt.legend()

# Exibir os dados filtrados
st.write(dados_filtrados_cluster_selecionado3)


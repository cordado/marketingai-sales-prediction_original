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

# Verifica se uma região foi selecionada
if region_selecionado:
    # Seleção do cluster
    cluster_selecionado = st.sidebar.selectbox('Selecione o cluster', options=dados_filtrados['cluster'].unique())
    
    # Verifica se um cluster foi selecionado
    if cluster_selecionado:
        # Filtra os dados pelo cluster selecionado
        dados_filtrados_cluster = dados_filtrados[dados_filtrados['cluster'] == cluster_selecionado]
        
        # Seleção da loja
        store_selecionado = st.sidebar.selectbox('Selecione a Loja', options=dados_filtrados_cluster['store'].unique())
        
        # Verifica se uma loja foi selecionada
        if store_selecionado:
            # Filtra os dados pela loja selecionada
            dados_filtrados_loja = dados_filtrados_cluster[dados_filtrados_cluster['store'] == store_selecionado]
            
            # Seleção do item
            item_selecionado = st.sidebar.selectbox('Selecione o Item', options=dados_filtrados_loja['item'].unique())

# Filtrar os dados
if region_selecionado and cluster_selecionado and store_selecionado and item_selecionado:
    dados_filtrados_cluster_selecionado2 = dados_filtrados[(dados_filtrados['cluster'] == cluster_selecionado) & (dados_filtrados['item'] == item_selecionado) & (dados_filtrados['store'] == store_selecionado)]
else:
    dados_filtrados_cluster_selecionado2 = dados_filtrados

# Agrupar os dados
dados_filtrados_cluster_selecionado3 = dados_filtrados_cluster_selecionado2.groupby(['year_month'])['SOMA'].sum().reset_index()

# Botão para resetar filtros
if st.sidebar.button('Resetar Filtros'):
    dados_filtrados_cluster_selecionado2 = dados_filtrados
    dados_filtrados_cluster_selecionado3 = dados_filtrados_cluster_selecionado2.groupby(['year_month'])['SOMA'].sum().reset_index()

# Preparar os dados para o Prophet
dados_prophet = dados_filtrados_cluster_selecionado3.rename(columns={'year_month': 'ds', 'SOMA': 'y'})

# Botão para executar a previsão
if st.button('Executar Previsão'):
    modelo = Prophet()
    modelo.fit(dados_prophet)
    futuro = modelo.make_future_dataframe(periods=20, freq='MS')
    previsao = modelo.predict(futuro)

    # Plotar os resultados
    fig = px.line(previsao, x='ds', y='yhat', title='Previsão com Prophet')
    st.plotly_chart(fig)

# Exibir os dados filtrados
st.write(previsao)

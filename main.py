import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Exibir os dados filtrados
st.markdown("# **DASHBOARD DA EMPRESA XXXX**")

# Carregar os dados
dados_frame1 = pd.read_csv('https://relacoesinstitucionais.com.br/Fotos/Temp/dados_csv_Boston_sem_outliers.csv')
dados_frame2 = pd.read_csv('https://relacoesinstitucionais.com.br/Fotos/Temp/dados_csv_New_York_sem_outliers.csv')
dados_frame3 = pd.read_csv('https://relacoesinstitucionais.com.br/Fotos/Temp/dados_csv_Philadelphia_sem_outliers.csv')

# Sidebar para filtros
st.sidebar.header('Escolher a região, cluster, loja e item')

# Seleção da região
region_selecionado = st.sidebar.selectbox('Selecione a Região', options=['Escolha uma opção', 'Boston', 'New_Yor', 'Philadelphia'])

# Carrega o DataFrame correto com base na região selecionada
if region_selecionado == 'Escolha uma opção':
    dados_filtrados = None
elif region_selecionado == 'Boston':
    dados_filtrados = dados_frame1
elif region_selecionado == 'New_Yor':
    dados_filtrados = dados_frame2
elif region_selecionado == 'Philadelphia':
    dados_filtrados = dados_frame3

# Exibir os dados filtrados, se houver
if dados_filtrados is not None:
    st.write(dados_filtrados)
else:
    st.write("Nenhuma região selecionada.")

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


# Botão para executar a previsão
if st.button('Executar Previsão'):
    modelo = Prophet(interval_width=0.95, daily_seasonality=False)
    modelo.fit(dados_prophet)
    futuro = modelo.make_future_dataframe(periods=20, freq='MS')
    previsao = modelo.predict(futuro)

    # Plotar os resultados
    fig = px.line(previsao, x='ds', y='yhat', title='Previsão com Prophet')
    st.plotly_chart(fig)

    

    df_real = pd.DataFrame(dados_prophet)
    df_real['ds'] = pd.to_datetime(df_real['ds'])
    previsao['ds'] = pd.to_datetime(previsao['ds'])

    # Filtrar df_real até 2015-12-01 e previsao igual ou acima de 2016-01-01
    df_real_filtered = df_real[df_real['ds'] <= '2015-12-01']
    previsao_filtered = previsao[previsao['ds'] >= '2016-01-01']

    # Concatenar os DataFrames filtrados
    df_merged = pd.concat([df_real_filtered, previsao_filtered[['ds', 'yhat']]], axis=0).reset_index(drop=True)

     # Plotar os valores reais vs preditos usando Matplotlib e Streamlit
    plt.figure(figsize=(10, 5))
    plt.plot(df_real_filtered['ds'], df_real_filtered['y'], label='Valor Real', marker='o')
    plt.plot(previsao_filtered['ds'], previsao_filtered['yhat'], label='Valor Predito', marker='x')
    
    # Adicionando título e rótulos
    plt.title('Comparação entre Valor Real e Valor Predito')
    plt.xlabel('Data')
    plt.ylabel('Valores')
    plt.legend()
    st.pyplot(plt)





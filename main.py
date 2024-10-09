import streamlit as st
import pandas as pd
from fbprophet import Prophet
import plotly.express as px

# Carregar os dados
dados_filtrados = pd.read_csv('https://relacoesinstitucionais.com.br/Fotos/Temp/base_mensal.csv')

# Sidebar para filtros
st.sidebar.header('Filtros')
item_selecionado = st.sidebar.selectbox('Selecione o Item', options=dados_filtrados['item'].unique())
store_selecionado = st.sidebar.selectbox('Selecione a Loja', options=dados_filtrados['store'].unique())

# Filtrar os dados
if item_selecionado and store_selecionado:
    dados_filtrados_cluster_selecionado2 = dados_filtrados[(dados_filtrados['item'] == item_selecionado) & (dados_filtrados['store'] == store_selecionado)]
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
    futuro = modelo.make_future_dataframe(periods=12, freq='M')
    previsao = modelo.predict(futuro)

    # Plotar os resultados
    fig = px.line(previsao, x='ds', y='yhat', title='Previsão com Prophet')
    st.plotly_chart(fig)

# Exibir os dados filtrados
st.write(dados_filtrados_cluster_selecionado3)

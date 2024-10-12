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
dados_frame = pd.read_csv('https://relacoesinstitucionais.com.br/Fotos/Temp/dados_boston_sem_outliers.csv')

# Sidebar para filtros
st.sidebar.header('Escolher o cluster, region e store')

# Seleção da região
cluster_selecionado = st.sidebar.selectbox('Selecione o Cluster', options=['Escolha uma opção', '0', '1', '2', '3', '4'])

# Carrega o DataFrame correto com base na região selecionada
if cluster_selecionado == 'Escolha uma opção':
    dados_frame_cluster  = dados_frame
else:
    dados_frame_cluster = dados_frame[dados_frame['Cluster'] == cluster_selecionado]

# Exibir os dados filtrados, se houver
if cluster_selecionado != 'Escolha uma opção':
    # Painel Geral dos Clusters
    final = dados_frame.groupby(['Cluster'])[['SOMA']].sum().reset_index()
    final['SOMA_TOTAL'] = final['SOMA'].sum()
    final['PERCENTUAL_SOMA'] = final['SOMA'] / final['SOMA_TOTAL']
    final['SOMA'] = final['SOMA'].apply(lambda x: f'R${x:,.2f}')
    final['SOMA_TOTAL'] = final['SOMA_TOTAL'].apply(lambda x: f'R${x:,.2f}')
    final['PERCENTUAL_SOMA'] = final['PERCENTUAL_SOMA'].apply(lambda x: f'{x * 100:.2f}%')
    final2 = dados_frame.groupby(['Cluster'])[['sales']].sum().reset_index()
    final2['sales_TOTAL'] = final2['sales'].sum()
    final2['PERCENTUAL_SALES'] = final2['sales'] / final2['sales_TOTAL']
    final2['sales'] = final2['sales'].apply(lambda x: f'{int(x):,}'.replace(',', '.'))
    final2['sales_TOTAL'] = final2['sales_TOTAL'].apply(lambda x: f'{int(x):,}'.replace(',', '.'))
    final2['PERCENTUAL_SALES'] = final2['PERCENTUAL_SALES'].apply(lambda x: f'{x * 100:.2f}%')
    final3 = dados_frame.groupby(['Cluster'])[['mean_price']].mean().reset_index()
    final3['mean_price_TOTAL'] = final3['mean_price'].sum()
    final3['PERCENTUAL_MEAN'] = final3['mean_price'] / final3['mean_price_TOTAL']
    final3['mean_price'] = final3['mean_price'].apply(lambda x: f'R${x:,.2f}')
    final3['mean_price_TOTAL'] = final3['mean_price_TOTAL'].apply(lambda x: f'R${x:,.2f}')
    final3['PERCENTUAL_MEAN'] = final3['PERCENTUAL_MEAN'].apply(lambda x: f'{x * 100:.2f}%')
    finais = pd.merge(final, final2, on='Cluster', how='outer')
    finais = pd.merge(finais, final3, on='Cluster', how='outer')
    st.dataframe(finais)
else:
    st.write("Nenhum cluster selecionado.")

# Seleção da região

regiao_selecionada = st.sidebar.selectbox('Selecione a região', options=dados_frame_cluster['region'].unique())

# Filtra os dados pela região selecionada
regiao_selecionada = dados_frame_cluster[dados_frame_cluster['region'] == regiao_selecionada]

# Seleção da loja
store_selecionado = st.sidebar.selectbox('Selecione a Loja', options=regiao_selecionada['store'].unique())

# Filtra os dados pela loja selecionada
dados_filtrados_loja = regiao_selecionada[regiao_selecionada['store'] == store_selecionado]

# Filtrar os dados
# dados_filtrados_cluster_selecionado2 = dados_filtrados[(dados_filtrados['Cluster'] == cluster_selecionado) & (dados_filtrados['item'] == item_selecionado) & (dados_filtrados['store'] == store_selecionado)]

# Agrupar os dados
dados_filtrados_cluster_selecionado3 = dados_filtrados_loja.groupby(['year_month'])['SOMA'].sum().reset_index()

# Botão para resetar filtros
if st.sidebar.button('Resetar Filtros'):
    dados_filtrados_cluster_selecionado3 = dados_frame.groupby(['year_month'])['SOMA'].sum().reset_index()

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





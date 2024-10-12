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

# Seleção do cluster
cluster_selecionado = st.sidebar.selectbox('Selecione o Cluster', options=['Escolha uma opção'] + list(dados_frame['Cluster'].unique().astype(str)))

# Carrega o DataFrame correto com base no cluster selecionado
if cluster_selecionado == 'Escolha uma opção':
    dados_frame_cluster = dados_frame
else:
    dados_frame_cluster = dados_frame[dados_frame['Cluster'] == int(cluster_selecionado)]

# Exibir os dados filtrados, se houver
if cluster_selecionado != 'Escolha uma opção':
    # Painel Geral dos Clusters
    final = dados_frame.groupby(['Cluster'])[['SOMA']].sum().reset_index()
    final['SOMA_TOTAL'] = final['SOMA'].sum()
    final['Percentual_Soma'] = final['SOMA'] / final['SOMA_TOTAL']
    final['SOMA'] = final['SOMA'].apply(lambda x: f'${x:,.2f}')
    final['SOMA_TOTAL'] = final['SOMA_TOTAL'].apply(lambda x: f'${x:,.2f}')
    final['Percentual_Soma'] = final['Percentual_Soma'].apply(lambda x: f'{x * 100:.2f}%')
    final2 = dados_frame.groupby(['Cluster'])[['sales']].sum().reset_index()
    final2['sales_TOTAL'] = final2['sales'].sum()
    final2['Percentual_Sales'] = final2['sales'] / final2['sales_TOTAL']
    final2['sales'] = final2['sales'].apply(lambda x: f'{int(x):,}'.replace(',', '.'))
    final2['sales_TOTAL'] = final2['sales_TOTAL'].apply(lambda x: f'{int(x):,}'.replace(',', '.'))
    final2['Percentual_Sales'] = final2['Percentual_Sales'].apply(lambda x: f'{x * 100:.2f}%')
    final3 = dados_frame.groupby(['Cluster'])[['mean_price']].mean().reset_index()
    final3['mean_price_TOTAL'] = final3['mean_price'].sum()
    final3['Percentual_Means'] = final3['mean_price'] / final3['mean_price_TOTAL']
    final3['mean_price'] = final3['mean_price'].apply(lambda x: f'${x:,.2f}')
    final3['mean_price_TOTAL'] = final3['mean_price_TOTAL'].apply(lambda x: f'${x:,.2f}')
    final3['Percentual_Means'] = final3['Percentual_Means'].apply(lambda x: f'{x * 100:.2f}%')
    finais = pd.merge(final, final2, on='Cluster', how='outer')
    finais = pd.merge(finais, final3, on='Cluster', how='outer')
    finais.set_index('Cluster', inplace=True)
    st.dataframe(finais[['SOMA', 'Percentual_Soma', 'sales', 'Percentual_Sales', 'mean_price', 'Percentual_Means']])
else:
    st.write("Nenhum cluster selecionado.")

# Seleção da região (após selecionar o cluster)
if cluster_selecionado != 'Escolha uma opção':
    regiao_escolhida = st.sidebar.selectbox('Selecione a região', options=['Escolha uma opção'] + list(dados_frame_cluster['region'].unique()))
    
    if regiao_escolhida != 'Escolha uma opção':
        regiao_filtrada = dados_frame_cluster[dados_frame_cluster['region'] == regiao_escolhida]
        
        # Seleção da loja (após selecionar a região)
        store_selecionado = st.sidebar.selectbox('Selecione a Loja', options=['Escolha uma opção'] + list(regiao_filtrada['store'].unique()))
        
        if store_selecionado != 'Escolha uma opção':
            dados_filtrados_loja = regiao_filtrada[regiao_filtrada['store'] == store_selecionado]
            dados_filtrados_cluster_selecionado3 = dados_filtrados_loja.groupby(['year_month'])['SOMA'].sum().reset_index()
            dados_filtrados_loja = dados_filtrados_loja[['year_month','item', 'region', 'store', 'sales', 'mean_price', 'SOMA']]
        else:
            st.write("Nenhuma loja selecionada.")
    else:
        st.write("Nenhuma região selecionada.")


    

# Botão para resetar filtros
if st.sidebar.button('Resetar Filtros'):
    dados_filtrados_cluster_selecionado3 = dados_frame.groupby(['year_month'])['SOMA'].sum().reset_index()


# Botão para executar a previsão
if st.button('Executar Previsão'):
    dados_prophet = dados_filtrados_cluster_selecionado3.rename(columns={'year_month': 'ds', 'SOMA': 'y'})
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
    plt.title(f'Comparação entre Valor Real e Valor Predito do Cluster {cluster_selecionado}, da região {regiao_filtrada} e da store {store_selecionado}')
    plt.xlabel('Data')
    plt.ylabel('Valores')
    plt.legend()
    st.pyplot(plt)


    
    # Assuming dados_filtrados_loja is your DataFrame
    top_5_stores = dados_filtrados_loja[['year_month','item','store', 'sales', 'mean_price', 'SOMA']].nlargest(10, 'SOMA')
    top_5_stores.set_index('year_month', inplace=True)

    # Display the DataFrame
    st.dataframe(top_5_stores)

    


    #





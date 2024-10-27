import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
import joblib
import requests
from sklearn.decomposition import PCA

# Carregar o algoritimo Kmeans já treinado

url = 'https://github.com/cordado/teste/raw/main/kmeans_treinado.pkl'

kmeans_TREINADO = joblib.load('kmeans_treinado.pkl')
response = requests.get(url)
open('kmeans_treinado.pkl', 'wb').write(response.content)
kmeans_TREINADO = joblib.load('kmeans_treinado.pkl')

# TELA DE APRESENTAÇÃO
st.markdown("# **Dashboard da Empresa MarketingAI**")

# SUBIR ARQUIVO ATÉ 200 MEGAS

uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file is None:
    st.write("Por favor, carregue um arquivo CSV.")
    
else:
    dados_frame = pd.read_csv(uploaded_file)
    dados_frame2 = dados_frame
   

    # Utilizar o LabelEncoder para trazer as mesmas configurações que utilizei para treinar o algoritimo
    
    label_encoder = LabelEncoder()
    dados_frame['year_month'] = pd.to_datetime(dados_frame['year_month'])
    dados_frame['region'] = label_encoder.fit_transform(dados_frame['region'])
    dados_frame['SOMA'] = dados_frame['sales'] * dados_frame['mean_price']
    dados_frame = dados_frame[dados_frame['year_month'] != '2011-01']
    dados_frame = dados_frame[dados_frame['sales'] != 0]
    dados_frame = dados_frame[~(dados_frame['mean_price'].isna() & (dados_frame['sales'] > 0))]
    
    dados_frame2['year_month'] = pd.to_datetime(dados_frame2['year_month'])
    dados_frame2['region'] = label_encoder.fit_transform(dados_frame2['region'])
    dados_frame2['SOMA'] = dados_frame2['sales'] * dados_frame2['mean_price']
    dados_frame2 = dados_frame2[dados_frame2['year_month'] != '2011-01']
    dados_frame2 = dados_frame2[dados_frame2['sales'] != 0]
    dados_frame2 = dados_frame2[~(dados_frame2['mean_price'].isna() & (dados_frame2['sales'] > 0))]
    
    # Padronizando os dados
    
    scaler = StandardScaler()
    dados_frame2 = scaler.fit_transform(dados_frame2[['sales', 'mean_price', 'region']])
    pca = PCA(n_components=2) 
    componentes_principais = pca.fit_transform(dados_frame2)
    df = pd.DataFrame(componentes_principais)
    dados_frame['Cluster'] = kmeans_TREINADO.predict(df)
    
    
    # Carregar os dados para outra variável e renomear as regions 
    
    dados_frame_completo_carregar_labels = dados_frame
    dados_frame['region'] = dados_frame['region'].replace({0: 'Boston', 1: 'New York', 2: 'Philadelphia'})
    
    # Sidebar para filtros
    st.sidebar.header('Escolher o cluster, region e store')
    
    # Seleção do cluster
    cluster_selecionado = st.sidebar.selectbox('Selecione o Cluster', options=['Escolha uma opção'] + list(dados_frame['Cluster'].unique().astype(str)))
    
    # Carrega o DataFrame correto com base no cluster selecionado

    if cluster_selecionado == 'Escolha uma opção':
        dados_frame_cluster = dados_frame
    else:
        dados_frame_cluster = dados_frame[dados_frame['Cluster'] == int(cluster_selecionado)]
    
    
    # Painel Geral dos Clusters para análise prévia

    # Objetivo é agrupar atráves do Cluster, a SOMA, SALES E MEAN_PRICE para demonstrar quais os Clusters possuem melhores índices.
    
    final = dados_frame.groupby(['Cluster'])[['SOMA']].mean().reset_index()
    final['SOMA_TOTAL'] = final['SOMA'].sum()
    final['Percentual_Soma'] = final['SOMA'] / final['SOMA_TOTAL']
    
    final2 = dados_frame.groupby(['Cluster'])[['sales']].mean().reset_index()
    final2['sales_TOTAL'] = final2['sales'].sum()
    final2['Percentual_Sales'] = final2['sales'] / final2['sales_TOTAL']
    
    final3 = dados_frame.groupby(['Cluster'])[['mean_price']].mean().reset_index()
    final3['mean_price_TOTAL'] = final3['mean_price'].sum()
    final3['Percentual_Means'] = final3['mean_price'] / final3['mean_price_TOTAL']
    
    finais_analise = pd.merge(final, final2, on='Cluster', how='outer')
    finais_analise = pd.merge(finais_analise, final3, on='Cluster', how='outer')
        
        
    soma_final = pd.DataFrame({'Cluster': ['Total'],'SOMA': [final['SOMA'].sum()],'SOMA_TOTAL': [final['SOMA'].sum()],'Percentual_Soma': ['100.00%'],'sales': [final2['sales'].sum()],'sales_TOTAL': [final2['sales'].sum()],'Percentual_Sales': ['100.00%'],'mean_price': [final3['mean_price'].sum()],
    'mean_price_TOTAL': [final3['mean_price'].sum()],'Percentual_Means': ['100.00%']})
    
    soma_final['SOMA'] = soma_final['SOMA'].apply(lambda x: f'${x:,.2f}')
    soma_final['sales'] = soma_final['sales'].apply(lambda x: f'{int(x):,}'.replace(',', '.'))
    soma_final['mean_price'] = soma_final['mean_price'].apply(lambda x: f'${x:,.2f}')
    
    final['SOMA'] = final['SOMA'].apply(lambda x: f'${x:,.2f}')
    final['SOMA_TOTAL'] = final['SOMA_TOTAL'].apply(lambda x: f'${x:,.2f}')
    final['Percentual_Soma'] = final['Percentual_Soma'].apply(lambda x: f'{x * 100:.2f}%')
    final2['sales'] = final2['sales'].apply(lambda x: f'{int(x):,}'.replace(',', '.'))
    final2['sales_TOTAL'] = final2['sales_TOTAL'].apply(lambda x: f'{int(x):,}'.replace(',', '.'))
    final2['Percentual_Sales'] = final2['Percentual_Sales'].apply(lambda x: f'{x * 100:.2f}%')
    final3['mean_price'] = final3['mean_price'].apply(lambda x: f'${x:,.2f}')
    final3['mean_price_TOTAL'] = final3['mean_price_TOTAL'].apply(lambda x: f'${x:,.2f}')
    final3['Percentual_Means'] = final3['Percentual_Means'].apply(lambda x: f'{x * 100:.2f}%')
    
    finais = pd.merge(final, final2, on='Cluster', how='outer')
    finais = pd.merge(finais, final3, on='Cluster', how='outer')
        
    finais = pd.concat([finais, soma_final], ignore_index=True)
        
    
           
    # Análise automática dos clusters, mesmo carregando um dataset diferente
    
    maior_soma = finais_analise.loc[finais_analise['SOMA'].idxmax()]
    maior_sales = finais_analise.loc[finais_analise['sales'].idxmax()]
    maior_mean_price = finais_analise.loc[finais_analise['mean_price'].idxmax()]
    
    mensagem1 = (f"O cluster {int(maior_soma['Cluster'])} possui a maior média da SOMA com representação de {maior_soma['Percentual_Soma']* 100:.2f}% sobre os outros Clusters.")
    mensagem2 = (f"O cluster {int(maior_sales['Cluster'])} possui a maior da SALES com representação de {maior_sales['Percentual_Sales']* 100:.2f}% sobre os outros Clusters.")
    mensagem3 = (f"O cluster {int(maior_mean_price['Cluster'])} possui a maior da MEAN_PRICE com representação de {maior_mean_price['Percentual_Means']* 100:.2f}% sobre os outros Clusters.")
    
    # Exibir a mensagem no Streamlit
    st.write(mensagem1)
    st.write(mensagem2)
    st.write(mensagem3)
    
    st.dataframe(finais[['Cluster', 'SOMA', 'Percentual_Soma', 'sales', 'Percentual_Sales', 'mean_price', 'Percentual_Means']].set_index('Cluster', drop=True))
        
    
    # Seleção da região (após selecionar o cluster)
    
    if cluster_selecionado != 'Escolha uma opção':
        regiao_escolhida = st.sidebar.selectbox('Selecione a region', options=['Escolha uma opção'] + list(dados_frame_cluster['region'].unique()))
        
        if regiao_escolhida != 'Escolha uma opção':
            regiao_filtrada = dados_frame_cluster[dados_frame_cluster['region'] == regiao_escolhida]
            
            # Seleção da loja (após selecionar a região)
            store_selecionado = st.sidebar.selectbox('Selecione a store', options=['Escolha uma opção'] + list(regiao_filtrada['store'].unique()))
            
            if store_selecionado != 'Escolha uma opção':
                dados_filtrados_loja = regiao_filtrada[regiao_filtrada['store'] == store_selecionado]
                dados_filtrados_cluster_selecionado3 = dados_filtrados_loja.groupby(['year_month'])['SOMA'].sum().reset_index()
                dados_filtrados_loja = dados_filtrados_loja[['year_month','item', 'region', 'store', 'sales', 'mean_price', 'SOMA']]

            ## BOTÃO PARA EXECUTAR PREVISAO COM O PROPHET
                
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
                    df_real_filtered = df_real[df_real['ds'] <= '2016-04-01']
                    previsao_filtered = previsao[previsao['ds'] > '2016-03-01']
                
                    # Concatenar os DataFrames filtrados
                    df_merged = pd.concat([df_real_filtered, previsao_filtered[['ds', 'yhat']]], axis=0).reset_index(drop=True)
                
                     # Plotar os valores reais vs preditos usando Matplotlib e Streamlit
                    plt.figure(figsize=(6, 3))
                    plt.plot(df_real_filtered['ds'], df_real_filtered['y'], label='Valor Real')
                    plt.plot(previsao_filtered['ds'], previsao_filtered['yhat'], label='Valor Predito')
                    
                    # Adicionando título e rótulos
                    plt.title(f'Previsão do Cluster {cluster_selecionado}, da região {regiao_escolhida} e da store {store_selecionado}')
                    plt.xlabel('Data')
                    plt.ylabel('Valores')
                    plt.legend()
                    st.pyplot(plt)
    
                    
                
                    
                    # Selecionando os 200 items que mais venderam entre 2011 e 2016
                    
                    top_200_itens = dados_filtrados_loja[['year_month','item','store', 'sales', 'mean_price', 'SOMA']].nlargest(200, 'SOMA')
                    
                    # Dos 200, as 10 que mais se repetem:
                    
                    top_10_rep = top_200_itens[['item']].value_counts().head(5)
    
                    
                    
                    top_10_stores = dados_filtrados_loja[['year_month','item','store', 'sales', 'mean_price', 'SOMA']].nlargest(10, 'SOMA') 
        
                    top_10_stores.set_index('year_month', inplace=True)
    
                    top_200_itens = dados_filtrados_loja[['year_month','item','store', 'sales', 'mean_price', 'SOMA']].nlargest(200, 'SOMA')
                    top_200_itens_2015_2016 = top_200_itens[top_200_itens['year_month'].dt.year.isin([2015, 2016])]
                    top_10_rep_2015_2016 = top_200_itens_2015_2016[['item']].value_counts().reset_index()
                    top_5_2015_16 = top_200_itens_2015_2016.value_counts().head(5)
                    
                     
                    
                    # Display the DataFrame
                    st.markdown("---")
                    
                    st.write(f'As dez maiores vendas do Cluster {cluster_selecionado}, da região {regiao_escolhida} e da store {store_selecionado}')
                    st.dataframe(top_10_stores)
                    
                    st.markdown("---")
                
                    st.write(f'Dos 200 items mais vendidos entre 2011 e inicío de 2016 do Cluster {cluster_selecionado}, da região {regiao_escolhida} e da store {store_selecionado} os items que mais se repetem:')
                    fig2, ax3 = plt.subplots()
                    top_10_rep.plot(kind='barh', ax=ax3)  # Use "barh" para gráfico de barras horizontal
                    for p in ax3.patches:
                        ax3.annotate(str(p.get_width()), (p.get_width(), p.get_y() + p.get_height() / 2.), ha='center', va='center', xytext=(10, 0), textcoords='offset points')
                    st.pyplot(fig2)
    
                    st.markdown("---")
                    st.write(f'Dos 200 items mais vendidos entre 2015 e inicío de 2016 do Cluster {cluster_selecionado}, da região {regiao_escolhida} e da store {store_selecionado} os items que mais se repetem:')
                    fig3, ax3 = plt.subplots()
                    top_10_rep_2015_2016_5 = top_10_rep_2015_2016.head(5)
                    top_10_rep_2015_2016_5.set_index('item', inplace=True)
                    top_10_rep_2015_2016_5.plot(kind='barh', ax=ax3)  # Use "barh" para gráfico de barras horizontal
                    for p in ax3.patches:
                        ax3.annotate(str(p.get_width()), (p.get_width(), p.get_y() + p.get_height() / 2.), ha='center', va='center', xytext=(10, 0), textcoords='offset points')
                    st.pyplot(fig3)
    
        
        
    
                    st.markdown("---")
    
                    st.write(f'Dos 200 items mais vendidos entre 2015 e inicío de 2016 do Cluster {cluster_selecionado}, da região {regiao_escolhida} e da store {store_selecionado} os items que mais se repetem e suas predições:')
                    grouped_dataframes = []
    
               
    
                    # Loop pelas primeiras 5 linhas do dataset 'top_10_rep_2015_2016' para retornar gráficos de cada item
                    for i in range(5):
                        # Obter o 'item' da linha atual
                        item = top_10_rep_2015_2016.iloc[i]['item']
                        
                        # Filtrar e agrupar o dataframe 'dados_filtrados_loja' pelo item selecionado
                        grouped_df = dados_filtrados_loja[dados_filtrados_loja['item'] == item].groupby(['item', 'year_month'])['SOMA'].sum().reset_index()
                        
                        # Adicionar o dataframe agrupado à lista
                        grouped_dataframes.append(grouped_df)
                        
                        # Preparar os dados para o Prophet
                        df_prophet = grouped_df[['year_month', 'SOMA']].rename(columns={'year_month': 'ds', 'SOMA': 'y'})
                        
                        # Converter a coluna 'ds' para datetime
                        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
                        
                        # Inicializar e ajustar o modelo Prophet
                        m = Prophet(interval_width=0.95, daily_seasonality=False)
                        m.fit(df_prophet)
                        
                        # Criar um dataframe para as previsões futuras
                        future = m.make_future_dataframe(periods=12, freq='MS')
                        
                        # Fazer previsões
                        forecast = m.predict(future)
                        
                        # Plotar os resultados usando Streamlit
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(df_prophet['ds'], df_prophet['y'], label='Valor Real', marker='o')
                        ax.plot(forecast['ds'], forecast['yhat'], label='Valor Predito', marker='x')
                        
                        # Adicionando título e rótulos
                        ax.set_title(f'Previsão para o item: {item}')
                        ax.set_xlabel('Data')
                        ax.set_ylabel('SOMA')
                        ax.legend()
                        
                        # Exibir o gráfico no Streamlit
                        st.pyplot(fig)
    
    
    
                    st.markdown("---")
                    st.markdown("## *Fim da Análise*")
                    
                    
    
                    
                    
            else:
                st.write("Nenhuma store selecionada.")
        else:
            st.write("Nenhuma region selecionada.")
    else:
        st.write("Por favor, selecione um CLUSTER para executar a previsão.")
    
    
        
    

        
    
           
        
        






# MarketingAI: Projeto de Análise Preditiva de Vendas

O projeto visa atender a MarketingAI, uma cadeia de shoppings, para realização de análises preditivas que resultem em aumento de vendas de produtos em suas diversas lojas.

O repositório se divide em diretórios 

Os requisitos são as bibliotecas: streamlit, pandas, scikit-learn, matplotlib, seaborn, joblib, Prophet, plotly_express e requests

A metodologia adotada foi a utilização de algoritimos de machine learning, para realizar a análise da base de dados da MarketingAI. O primeiro algoritimo utilizado foi o Kmeans, com o intuito de "clusterizar" de modo automático e trazer novos insights para análise. Outro algoritimo utilizado foi o Prophet para predizer as vendas futuras em um período de 1 ano, utilizando a base de dados de períodos anteriores.

Os resultados alcançados mostram que a clusterização resultou na separação em 5 grupos. Dois grupos trouxeram uma separação evidente: 1 grupo com produtos de alto valor (premium) mas que possuem poucas vendas. Outro grupo com valor baixo, mas com alta quantidade de vendas. Os outros 3 grupos ficaram mais parecidos, com valor médio de produtos e de vendas. 

O deploy foi realizado no link: XXXXX , com a criação de dashboard interativo, que possibilita a seleção de cada cluster. Cada cluster tem uma breve análise, demontrando suas características gerais. É possível selecionar a região e a store. Após a seleção, é apresentado a predição daquela loja em relação ao tipo de produtos. É demonstrado os 10 produtos mais vendidos de todo o período e nos últimos dois anos (2015 e 2016). É feita a predição desses produtos para o próximo ano, demonstrando sua tendência.

Autor: Mateus Felipe Picosque

Agradecimento ao Prof. Eduardo que sempre está presente para seus alunos e fornecendo um curso de alta qualidade.

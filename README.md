# 🎬🤖 Dataflix - Insights de Hollywood: Prevendo notas IMDb com Machine Learning

## 🎯 Objetivo do Projeto
Este projeto de Ciências de Dados analisa filmes do IMDb para entender os fatores que explicam altas avaliações e prever notas futuras. Além do modelo preditivo, o trabalho oferece recomendações estratégicas para identificar gêneros com maior potencial de sucesso

## 🛠️ Ferramentas e Bibliotecas
Linguagem: Python
Bibliotecas Principais: Pandas (manipulação de dados), Matplotlib e Seaborn (visualização), Scikit-learn (modelagem de machine learning), LightGBM.
APIs: OMDb API (para enriquecimento de dados).

## 📂 Estrutura do Projeto
```
├── LH_CD_ANAPAULA.ipynb      # Notebook principal com todo o pipeline
├── desafio_indicium_imdb.csv      # O conjunto de dados inicial
├── desafio_indicium_imdb_completo.csv      # O conjunto de dados limpo e pré-processado
├── filmes_para_buscar.csv e filmes_buscar_omdb_atualizado.csv      #  Arquivos intermediários do processo de busca na API
├── requirements.txt      # Lista de pacotes e versões para reprodutibilidade do ambiente.
├── modelo_imdb.pkl      # O modelo preditivo final treinado e salvo
```

## 📊 Análise Exploratória de Dados (EDA) - Principais Descobertas
Nossa análise exploratória foi uma jornada para entender a "personalidade" dos filmes de sucesso. 
Começamos com a hipótese de que a maioria dos filmes aclamados teria notas altas e concentradas, o que foi confirmado por um histograma que mostrou a maior parte das avaliações entre 7.6 e 8.5.

A partir daí, mergulhamos nos fatores que poderiam influenciar essa nota. A nossa hipótese principal era que o gênero seria um fator-chave. A análise com box plots revelou um cenário fascinante:

Gêneros como Film-Noir, War e Mistery apresentaram as maiores medianas, sendo apostas consistentes para alta aclamação.

Os gêneros Drama e Crime, apesar de terem uma variação maior de notas, foram os que produziram os filmes com as avaliações mais altas do dataset (os outliers no topo do gráfico), revelando uma natureza de "alto risco, alta recompensa".

Investigamos também o fator humano, levantando a hipótese de que diretores e atores de renome impactariam o sucesso. 

## ❓ Perguntas de Negócio
- Qual filme você recomendaria para uma pessoa que você não conhece?

Baseado nos dados, a recomendação mais segura seria um filme que combine aclamação crítica com imensa popularidade. Um filme como "Forrest Gump" (Forrest Gump - O Contador de Histórias) seria uma excelente escolha. Ele pertence ao gênero Drama, que é o mais prevalente e um dos mais bem avaliados do dataset, e possui um número altíssimo de votos, indicando que agrada a uma vasta gama de perfis de público.

- Quais são os principais fatores que estão relacionados com alta expectativa de faturamento de um filme?

Nossa análise de correlação mostrou que o fator mais forte ligado ao faturamento (Gross) é a popularidade e o engajamento do público, representados pela coluna No_of_Votes (coeficiente de 0.55). Portanto, um filme que gera muita discussão e atrai muitas pessoas para votar (o que geralmente está ligado a grandes campanhas de marketing e a gêneros de apelo popular como Ação e Aventura) tem a maior expectativa de faturamento.

- Quais insights podem ser tirados com a coluna Overview? É possível inferir o gênero do filme a partir dessa coluna?

A coluna Overview é uma mina de ouro de dados textuais. Com técnicas de Processamento de Linguagem Natural (NLP), poderíamos extrair insights profundos, como identificar os temas e tópicos mais comuns em filmes de sucesso (ex: "vingança", "jornada do herói", "redenção") através de modelagem de tópicos. E sim, é totalmente possível inferir o gênero a partir dessa coluna. O processo seria converter as sinopses em vetores numéricos (usando TF-IDF, por exemplo) e treinar um modelo de classificação multi-rótulo para prever as categorias de gênero com base nas palavras e no contexto presentes no texto.
Comprovamos que diretores como Stanley Kubrick e atores como James Stewart não só apareciam com frequência, mas também estavam associados a notas médias altíssimas, validando a ideia de que "star power" é um fator preditivo real.

## 🧠 Modelagem Preditiva
Para prever a nota do IMDb, abordamos o problema como uma tarefa de Regressão, pois nosso alvo (IMDB_Rating) é um valor numérico contínuo.

As variáveis utilizadas foram uma combinação de dados brutos e, principalmente, de transformações que criamos (engenharia de features):

Numéricas: Meta_score, No_of_Votes, Released_Year, Runtime e Gross.

Categóricas Transformadas:

Gênero e Certificado: Foram transformados usando One-Hot Encoding para criar colunas binárias (ex: Genre_Drama, Certificate_Conteudo Adulto), permitindo que o modelo entendesse a presença ou ausência de cada categoria.

Diretor e Atores: Para evitar a criação de centenas de colunas, criamos features mais inteligentes:

diretor_top_10 e ator_top_15: Flags binárias que capturam o "star power".

diretor_media_nota: Uma feature de Target Encoding que representa a reputação do diretor por sua nota média histórica, que se provou ser a variável mais poderosa de todas.

O modelo que melhor se aproximou dos dados foi o Random Forest Regressor, especialmente após a otimização de hiperparâmetros.

Prós: É um modelo robusto, que lida bem com relações não-lineares entre as variáveis e é menos sensível a outliers. Além disso, nos permite analisar a importância das features, o que foi crucial para validar nossos insights.

Contras: Pode ser computacionalmente mais "caro" que modelos simples e, se não for bem ajustado, pode tender ao overfitting.

As medidas de performance escolhidas foram o RMSE e o R². 
O modelo final alcançou um RMSE de 0.1438 e R² de 0.6850, indicando boa capacidade preditiva.
O RMSE é excelente por ser de fácil interpretação (representa o erro médio na mesma escala da nota do IMDb), enquanto o R² nos deu uma visão clara do poder explicativo geral do modelo (o quanto ele consegue "entender" a variação das notas).


## 🔮 Previsão para 'The Shawshank Redemption'
Previsão para "The Shawshank Redemption": O modelo treinado previu uma nota de 8.75 para o filme, um valor próximo da sua nota real (9.3).

## ▶️ Como Rodar
1. Clone este repositório:
   ```bash
   git clone https://github.com/anapaulads/Dataflix-Insights-de-Hollywood.git
   ```
2. Instale as dependências (recomenda-se uso de ambiente virtual):
   ```bash
   pip install -r requirements.txt
   ```
3. Abra o notebook `LH_CD_ANAPAULA.ipynb` e execute as células sequencialmente.
   ``` 
5. Como Usar o Modelo: O modelo final foi salvo no arquivo modelo_imdb.pkl. Ele pode ser carregado com a biblioteca pickle e utilizado para fazer novas previsões em dados de filmes que sigam a mesma estrutura do treino.

## 🤝🏾 Contribuições
Sugestões, melhorias e novas ideias são bem-vindas!  
Sinta-se à vontade para abrir issues ou pull requests.

## ✍🏾 Autoria
Projeto desenvolvido por Ana Paula Dias, como um desafio proposto de Ciência de Dados. 

---

> Este projeto demonstra habilidades em análise de dados, engenharia de variáveis, modelagem preditiva, explicabilidade e comunicação de resultados.

#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk

from sklearn.preprocessing import (KBinsDiscretizer, LabelEncoder,
                                   OneHotEncoder, StandardScaler,FunctionTransformer)
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# In[5]:


countries.shape


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[6]:


# Sua análise começa aqui.
#print (countries.dtypes.unique().size)
#print (countries.dtypes.unique())

#countries['Pop_density'] = pd.to_numeric(countries['Pop_density'], errors='coerce')
#countries['Pop_density'].astype(float)
#countries['Pop_density'] = countries.Pop_density.astype(float)

colunms_para_inteiro = ['Pop_density', 'Coastline_ratio', 'Net_migration', 'Infant_mortality', 'Literacy', 'Phones_per_1000', 'Arable', 'Crops', 'Other', 'Climate','Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service']


for i in colunms_para_inteiro:
    #print (i)
    #print ('So I said, "You don\'t know me! You\'ll never understand me!"')
    #print (countries[i])
    countries[i] = countries[i].str.replace(',', '.').astype(float)
    



countries['Country'] = countries.Country.str.strip()
countries['Region'] = countries.Region.str.strip()


 


# In[7]:


print (countries.dtypes)
countries.head(5)


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[8]:


def q1():
    # Retorne aqui o resultado da questão 1.
    region = countries['Region'].unique().tolist()
    region.sort()
    print (region)
    #print ( np.sort(countries['Region'].unique()).tolist())
    return region




    


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[9]:


def q2():
    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    #print (est)
    discretizando = est.fit_transform(np.array(countries["Pop_density"]).reshape(-1, 1))
    #print (discretizando)
    # obtendo 90º percentil
    percentil90 = np.quantile(discretizando, 0.9)
    #print (percentil90)
    
    resposta = len(discretizando[discretizando > percentil90])
    #print (resposta)
    
    return int(resposta)
    
    
    # Retorne aqui o resultado da questão 2.
    


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[10]:


def q3():
    
    #print (countries['Region'].unique())
    
    #print (countries['Climate'].unique())
    
    
    #label_encoder = LabelEncoder()
    #Regioes = label_encoder.fit_transform(countries["Region"])
    #climate = label_encoder.fit_transform(countries["Climate"])
    #countries["Climate"] = label_encoder.fit_transform(countries["Climate"])
    #countries["Region"] = label_encoder.fit_transform(countries["Region"])
    #diferentes_climas = (countries["Climate"].nunique())
    #diferentes_regioes = (countries["Region"].nunique())
    #print (diferentes_climas + diferentes_regioes)
    #return int(diferentes_climas + diferentes_regioes)
    
    #one-hot_encoding
    
       # Retorne aqui o resultado da questão 3.
    
    # objeto para converter categórico para numérico
    le = preprocessing.LabelEncoder()
    
    # objeto para aplicar oneHot
    enc = OneHotEncoder(handle_unknown='ignore')
    
    # variável temporária
    data = countries[["Region", "Climate"]]
    
    data.fillna({"Climate": 0}, inplace=True)
    
    # transformando colunas region em numérica
    data["Region"] = le.fit_transform(data["Region"])
    
    # aplicando o oneHotEncoding
    values = enc.fit_transform(data).toarray()
    
    # Pegando os nomes das novas colunas
    colNames = list(enc.get_feature_names(["Region", "Climate"]))
    
    # criando o dataframe com os novos dados
    newData = pd.DataFrame(values, columns=colNames)
     
    # obtendo o número de novas colunas adicionadas
    colunas = int(newData.shape[1])
    print(colunas)
    
    
    #getdummies
    
    resultado_get_dummies = pd.get_dummies(countries[['Region', 'Climate']].fillna('NaN'))
    #print (resultado_get_dummies)

    resultado = int(resultado_get_dummies.shape[1])
    #print (resultado)
    
    return resultado

    
    
    
    # Retorne aqui o resultado da questão 3.
    


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[11]:


#deixar apenas variaveis numericas(int64 e float64)
df_col = countries.columns.drop(['Country', 'Region'])
df_num = countries[df_col].astype(float)
print (df_num)



test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]

row_test_country = pd.DataFrame([test_country], columns = new_column_names)
row_test_country.info()


# In[12]:


def q4():
    
    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("std_scaler", StandardScaler())])    
    
    #print (num_pipeline)
    test_country_num = row_test_country.drop(['Country', 'Region'], axis = 1)
    #print (test_country_num)
    test_country_cat = row_test_country[['Country', 'Region']]
    #print (test_country_cat)
    df_num_fit = num_pipeline.fit(df_num)
    print (df_num_fit)
    test_cty_transf = num_pipeline.transform(test_country_num)
    print (test_cty_transf)
    test_cty_transf_df = pd.DataFrame(test_cty_transf, columns = df_num.columns)
    print (test_cty_transf_df)
    result = float(test_cty_transf_df['Arable'].round(3))
    print (result)
    return result


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[13]:


def q5():
    

    #quartis
    Q1 = countries['Net_migration'].quantile(q= 0.25)
    Q3 = countries['Net_migration'].quantile(q= 0.75)

    # Intervalo interquartil:
    IQR = Q3 - Q1

   # Limite inferior:
    lim_max = (countries['Net_migration'] < (Q1 - 1.5 * IQR)).sum()
    # Limite superior:
    lim_min = (countries['Net_migration'] > (Q3 + 1.5 * IQR)).sum()
    
    # Número de outliers:
    return int(lim_max), int(lim_min),False
q5()


# In[14]:


#Visualização utilizando histograma:
#sns.distplot(countries['Net_migration'], color='m');


# In[15]:


#Visualização utilizando boxplot:
ax = sns.boxplot(countries['Net_migration'], orient='vertical',color='r');


# In[16]:



#Tratar outliers:

#Podemos eliminá-los da nossa amostra;
#Podemos analisá-los de forma separada,
#Podemos realizar alguma transformação matemática para reduzir a variação dos dados.
#O intervalo interquartil é uma medida de dispersão utilizada em estatística descritiva. Seu cálculo se da pela subtração do 3° quartil pelo 1° quartil.

#POdemos considerar como outliers os valores menores que Q1-1,5*IQR ou valores maiores que Q3+1,5*IQR.


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[23]:


def q6():
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    
    #print (categories)

    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    print(newsgroup)
    
    count_vectorizer = CountVectorizer()

    newsgroups_counts = count_vectorizer.fit_transform(newsgroup.data)
    
    #print (newsgroups_counts)
    #print (count_vectorizer.vocabulary_)
    #print(int(newsgroups_counts[:, count_vectorizer.vocabulary_["phone"]].toarray().sum()))
    
    
    return int(newsgroups_counts[:, count_vectorizer.vocabulary_["phone"]].toarray().sum())

    


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[28]:


tfidf_vectorizer = TfidfVectorizer()

categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']

newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    

tfidf_vectorizer.fit(newsgroup.data)

newsgroups_tfidf_vectorized = tfidf_vectorizer.transform(newsgroup.data)


# In[29]:


def q7():
    
    idf_value = newsgroups_tfidf_vectorized[:, tfidf_vectorizer.vocabulary_["phone"]].toarray().sum()
    
    return float(round(idf_value, 3))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns


# In[2]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


athletes = pd.read_csv("athletes.csv")


# In[4]:


def get_sample(df, col_name, n=100, seed=42):
   
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
athletes.head()



# In[6]:


athletes.shape


# In[7]:


#athletes.hist()


# In[8]:


aux = get_sample(athletes, 'height', 3000)
aux.sort_values()


# In[9]:


aux.shape


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[10]:


def q1():
    stats, p_value = sct.shapiro(aux)
    return bool(p_value > 0.05)
    
q1()    


# In[11]:


#athletes.height.hist(bins = 25)


# In[12]:



#sct.probplot(athletes.height, dist ='norm', plot =plt)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[13]:


def q2():
    stats, p_value = sct.jarque_bera(aux)
    return bool(p_value > 0.05)
    
q2()    


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[14]:


sample_weight = get_sample(athletes, 'weight', 3000)


# In[15]:


#?sct.normaltest


# In[16]:


def q3():
    stats, p_value = sct.normaltest(sample_weight)
    return bool(p_value > 0.05)
    
q3()    


# In[17]:


#athletes.weight.hist(bins = 25)


# In[18]:


#sns.boxplot(athletes.weight)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[19]:


sample_weight_q4 = get_sample(athletes, 'weight', 3000)


# In[20]:


aux_q4 = np.log(sample_weight_q4)


# In[21]:


aux_q4


# In[22]:


def q4():
    stats, p_value = sct.normaltest(aux_q4)
    return bool(p_value > 0.05)
    
q4()    


# In[23]:


#aux_q4.hist(bins = 25)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[24]:


nacionalidade_bra = ['BRA']
s_bra = athletes[athletes.nationality.isin(nacionalidade_bra)]
nacionalidade_usa = ['USA']
s_usa = athletes[athletes.nationality.isin(nacionalidade_usa)]
nacionalidade_can = ['CAN']
s_can = athletes[athletes.nationality.isin(nacionalidade_can)]



# In[25]:


def q5():
    aux_q5 = sct.ttest_ind(s_bra.height, s_usa.height, equal_var = False, nan_policy = 'omit' )
    stats_q5, p_value_q5 = aux_q5
    return bool(p_value_q5 > 0.05)
q5()


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[33]:


def q6():
    aux_q6 = sct.ttest_ind(s_bra.height, s_can.height, equal_var = False, nan_policy = 'omit' )
    stats_q6, p_value_q6 = aux_q6
    return bool(p_value_q6 > 0.05)
q6()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[36]:


def q7():
    ttest=sct.ttest_ind(s_usa.height, s_can.height, equal_var=False, nan_policy='omit')
    
    return float(ttest.pvalue.round(8))
q7()


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?

# In[ ]:





# In[ ]:





# In[ ]:





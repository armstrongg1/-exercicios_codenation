#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[16]:


import pandas as pd
import numpy as np


# In[17]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[18]:



black_friday.head(3)


# In[19]:


black_friday['Age'].value_counts()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[20]:



def q1():
    return black_friday.shape
    
q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[21]:


def q2():
    return len(black_friday[(black_friday['Age'] == '26-35') & (black_friday['Gender'] == 'F')])
q2()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[22]:


def q3():
    return black_friday['User_ID'].nunique()
q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[23]:


def q4():
    return black_friday.dtypes.nunique()
q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[24]:


def q5():
    return 1 - len(black_friday.dropna()) / len(black_friday)
q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[25]:



def q6():
    return int(black_friday.isnull().sum().sort_values(ascending = False)[0])
q6()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[26]:



def q7():
    return black_friday['Product_Category_3'].value_counts().index[0]
q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[27]:


def q8():
    return ((black_friday['Purchase'] - black_friday['Purchase'].min())/(black_friday['Purchase'].max() - black_friday['Purchase'].min())).mean()
q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[28]:


def q9():
    lista = ((black_friday['Purchase'] - black_friday['Purchase'].mean())/black_friday['Purchase'].std())
    return int(((lista < 1) & (lista > -1)).sum())
q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[30]:


def q10():
    # Retorne aqui o resultado da questão 10.
    category_2 = black_friday[black_friday['Product_Category_2'].isna()]
    category_3 = category_2[category_2['Product_Category_3'].isna()]
    return category_2.shape == category_3.shape
q10()


# In[ ]:





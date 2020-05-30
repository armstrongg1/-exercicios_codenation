# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:28:31 2020

@author: Armstrong
"""
#modelo deseja prever as notasa de matematica. A variavel NOTA_MATEMATICA é minha variavel dependentes. Tenho que escolher as 
#melhorer variveis independentes que influciam na nota de matematica, ou seja, minhas variaveis independentes.
#o modelo que aqui se aplica é o de regressão. São vários os modelos de regressão que existem (linear,randonforest,etc)


#4. Underfitting e Overfitting
#Quando usamos variáveis explicativas desnecessárias, isso pode levar ao overfitting. Overfitting significa que nosso algoritmo funciona bem no conjunto de treinamento, mas não consegue ter um desempenho melhor nos conjuntos de teste. Também é conhecido como problema de alta variância.


import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
#from sklearn.preprocessing import LabelEncoder
#import random
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from sklearn.ensemble import GradientBoostingClassifier

#carrega dados
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")


cons_train_inicial = pd.DataFrame({'colunas' : train.columns,
                    'tipo': train.dtypes,
                    'missing' : train.isna().sum(),
                    'size' : train.shape[0],
                    'unicos' : train.nunique()})

cons_train_inicial['percentual'] =  round(cons_train_inicial['missing'] / cons_train_inicial['size'],2)



#variaveis strings para numero. nao usei o getdummies pq ele cria outras inumeras variaveis
LE = LabelEncoder()

test['Q001_numero'] = LE.fit_transform(test['Q001'])
train['Q001_numero'] = LE.fit_transform(train['Q001'])

test['Q002_numero'] = LE.fit_transform(test['Q002'])
train['Q002_numero'] = LE.fit_transform(train['Q002'])

test['Q006_numero'] = LE.fit_transform(test['Q006'])
train['Q006_numero'] = LE.fit_transform(train['Q006'])

test['Q0024_numero'] = LE.fit_transform(test['Q024'])
train['Q0024_numero'] = LE.fit_transform(train['Q024'])

test['Q0025_numero'] = LE.fit_transform(test['Q025'])
train['Q0025_numero'] = LE.fit_transform(train['Q025'])

test['Q0047_numero'] = LE.fit_transform(test['Q047'])
train['Q0047_numero'] = LE.fit_transform(train['Q047'])



test['CO_PROVA_CH_numero'] = LE.fit_transform(test['CO_PROVA_CH'])
train['CO_PROVA_CH_numero'] = LE.fit_transform(train['CO_PROVA_CH'])

test['CO_PROVA_LC_numero'] = LE.fit_transform(test['CO_PROVA_LC'])
train['CO_PROVA_LC_numero'] = LE.fit_transform(train['CO_PROVA_LC'])

test['CO_PROVA_MT_numero'] = LE.fit_transform(test['CO_PROVA_MT'])
train['CO_PROVA_MT_numero'] = LE.fit_transform(train['CO_PROVA_MT'])

test['CO_PROVA_CN_numero'] = LE.fit_transform(test['CO_PROVA_CN'])
train['CO_PROVA_CN_numero'] = LE.fit_transform(train['CO_PROVA_CN'])



#pega o valor inicial de nuincricao para jogar na planilha de resposta final
final = test.filter(['NU_INSCRICAO'])

#cria as variaves F e M no dataframe para sexo
dummy = pd.get_dummies(test['TP_SEXO'])
test = pd.concat([test,dummy],axis=1)

dummy = pd.get_dummies(train['TP_SEXO'])
train = pd.concat([train,dummy],axis=1)


#test_filtrada = test.filter(["NU_NOTA_CN","NU_NOTA_LC","NU_NOTA_CH","NU_NOTA_REDACAO"])

test_filtrada = test.filter(['F','M','NU_NOTA_REDACAO',
                  'NU_NOTA_CN',
                  'NU_NOTA_LC',
                  'NU_NOTA_CH',
                  'NU_IDADE',                  
                  'TP_ESCOLA',
                  'CO_UF_RESIDENCIA',
                  'TP_ANO_CONCLUIU',
                  'TP_LINGUA',
                  'NU_NOTA_COMP1',
                  'NU_NOTA_COMP2', 
                  'NU_NOTA_COMP3', 
                  'NU_NOTA_COMP4', 
                  'NU_NOTA_COMP5',
                  'Q001_numero',
                  'Q002_numero',
                  'Q006_numero',
                  'Q0024_numero',
                  'Q0025_numero',             
                  'Q0047_numero',
                  #'Q0027_numero',             
                  'CO_PROVA_CH_numero',
                  'CO_PROVA_LC_numero',
                  'CO_PROVA_MT_numero',
                  'CO_PROVA_CN_numero',
                  'TP_STATUS_REDACAO',
                  'TP_NACIONALIDADE',  
                  'TP_PRESENCA_CN',                
                   'TP_PRESENCA_CH',
                   'TP_PRESENCA_LC',
                   'IN_BAIXA_VISAO',
                   'IN_CEGUEIRA',
                   'IN_SURDEZ',
                   'IN_DISLEXIA',             
                   'IN_DISCALCULIA',
                   'IN_SABATISTA',
                   'IN_GESTANTE',
                   'IN_IDOSO',
                   'TP_ST_CONCLUSAO'])


cons_teste = pd.DataFrame({'colunas' : test_filtrada.columns,
                    'tipo': test_filtrada.dtypes,
                    'missing' : test_filtrada.isna().sum(),
                    'size' : test_filtrada.shape[0],
                    'unicos' : test_filtrada.nunique()})

cons_teste['percentual'] =  round(cons_teste['missing'] / cons_teste['size'],2)


#test_filtrada = test_filtrada.fillna(test_filtrada.mean())
#0 nas notas
test_filtrada = test_filtrada.fillna(0)

#df['Item_Weight'] = df['Item_Weight'].fillna((df['Item_Weight'].mean()))

#treino_filtrada = train.filter(["NU_NOTA_CN","NU_NOTA_LC","NU_NOTA_MT","NU_NOTA_CH","NU_NOTA_REDACAO"])




treino_filtrada = train.filter(['F','M','NU_NOTA_MT','NU_NOTA_REDACAO',
                  'NU_NOTA_CN',
                  'NU_NOTA_LC',
                  'NU_NOTA_CH',
                  'NU_IDADE',                  
                  'TP_ESCOLA',
                  'CO_UF_RESIDENCIA',
                  'TP_ANO_CONCLUIU',
                  'TP_LINGUA',
                  'NU_NOTA_COMP1',
                  'NU_NOTA_COMP2', 
                  'NU_NOTA_COMP3', 
                  'NU_NOTA_COMP4', 
                  'NU_NOTA_COMP5',
                  'Q001_numero',
                  'Q002_numero',
                  'Q006_numero',
                  'Q0024_numero',
                  'Q0025_numero',             
                  'Q0047_numero',                         
                  'CO_PROVA_CH_numero',
                  'CO_PROVA_LC_numero',
                  'CO_PROVA_MT_numero',
                  'CO_PROVA_CN_numero',
                  'TP_STATUS_REDACAO',
                  'TP_NACIONALIDADE',  
                  'TP_PRESENCA_CN',                
                   'TP_PRESENCA_CH',
                   'TP_PRESENCA_LC',
                   'IN_BAIXA_VISAO',
                   'IN_CEGUEIRA',
                   'IN_SURDEZ',
                   'IN_DISLEXIA',             
                   'IN_DISCALCULIA',
                   'IN_SABATISTA',
                   'IN_GESTANTE',
                   'IN_IDOSO',
                   'TP_ST_CONCLUSAO'])



cons_train = pd.DataFrame({'colunas' : treino_filtrada.columns,
                    'tipo': treino_filtrada.dtypes,
                    'missing' : treino_filtrada.isna().sum(),
                    'size' : treino_filtrada.shape[0],
                    'unicos' : treino_filtrada.nunique()})

cons_train['percentual'] =  round(cons_train['missing'] / cons_train['size'],2)

#treino_filtrada = treino_filtrada.fillna(treino_filtrada.mean())
#0 nas notas
treino_filtrada = treino_filtrada.fillna(0)












atributos_previstos = ['NU_NOTA_MT']
#atributos_que_incluenciam = ["NU_NOTA_CN","NU_NOTA_LC","NU_NOTA_CH","NU_NOTA_REDACAO"]

atributos_que_incluenciam = ['F','M','NU_NOTA_REDACAO',
                  'NU_NOTA_CN',
                  'NU_NOTA_LC',
                  'NU_NOTA_CH',
                  'NU_IDADE',                  
                  'TP_ESCOLA',
                  'CO_UF_RESIDENCIA',
                  'TP_ANO_CONCLUIU',
                  'TP_LINGUA',
                  'NU_NOTA_COMP1',
                  'NU_NOTA_COMP2', 
                  'NU_NOTA_COMP3', 
                  'NU_NOTA_COMP4', 
                  'NU_NOTA_COMP5',
                  'Q001_numero',
                  'Q002_numero',
                  'Q006_numero',
                  'Q0024_numero',
                  'Q0025_numero',             
                  'Q0047_numero',                        
                  'CO_PROVA_CH_numero',
                  'CO_PROVA_LC_numero',
                  'CO_PROVA_MT_numero',
                  'CO_PROVA_CN_numero',
                  'TP_STATUS_REDACAO',
                  'TP_NACIONALIDADE',  
                  'TP_PRESENCA_CN',                
                   'TP_PRESENCA_CH',
                   'TP_PRESENCA_LC',
                   'IN_BAIXA_VISAO',
                   'IN_CEGUEIRA',
                   'IN_SURDEZ',
                   'IN_DISLEXIA',             
                   'IN_DISCALCULIA',
                   'IN_SABATISTA',
                   'IN_GESTANTE',
                   'IN_IDOSO',
                   'TP_ST_CONCLUSAO']




X = treino_filtrada[atributos_que_incluenciam]
Y = treino_filtrada[atributos_previstos]

Z = test_filtrada[atributos_que_incluenciam]




X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

rf = RandomForestRegressor(n_estimators=1000,random_state = 42)

rf.fit(X_train, Y_train)

print (rf.score(X_train, Y_train))


rf.fit(X_test,Y_test)
print (rf.score(X_test,Y_test))



#rodando modelo
pred = rf.predict(Z)


predictions = pd.DataFrame(pred)


final['NU_NOTA_MT'] = predictions


#exportando o resultado
final.to_csv('answer.csv')


#autoML
#print ('comecando....')

#from tpot import TPOTRegressor

#tp = TPOTRegressor(verbosity = 2, scoring = 'neg_median_absolute_error')

#tp.fit(X_train, Y_train)












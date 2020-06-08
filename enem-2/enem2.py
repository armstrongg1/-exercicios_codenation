# -*- coding: utf-8 -*-
"""
Created on Sun May 31 09:22:03 2020
#uso o Spyder e não Jupyter notebook

@author: Armstrong
"""



import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor




#carrega dados
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")


#pega o valor inicial de nuincricao para jogar na planilha de resposta final
final = test.filter(['NU_INSCRICAO'])

#variavel que queremos - Já preencho com zero ( vou fazer isso pras outras notas também )
Y = train.filter(['NU_NOTA_MT']).fillna(0)

#acrescentar tp_cor_raca

#filtra variaveis que vão influenciar no modelo ( verifiquei pela correlação em outro script )
test = test.filter(['TP_COR_RACA','TP_DEPENDENCIA_ADM_ESC','TP_SEXO','NU_NOTA_REDACAO',
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
                  'Q001',
                  'Q002',
                  'Q006',
                  'Q024',
                  'Q025',             
                  'Q047',                  
                  'CO_PROVA_CH',
                  'CO_PROVA_LC',
                  'CO_PROVA_MT',
                  'CO_PROVA_CN',
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

#filtra variaveis que vão influenciar no modelo
train = train.filter(['TP_COR_RACA','TP_DEPENDENCIA_ADM_ESC','TP_SEXO','NU_NOTA_REDACAO',
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
                  'Q001',
                  'Q002',
                  'Q006',
                  'Q024',
                  'Q025',             
                  'Q047',                  
                  'CO_PROVA_CH',
                  'CO_PROVA_LC',
                  'CO_PROVA_MT',
                  'CO_PROVA_CN',
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

#identificando miss values
cons_train_inicial = pd.DataFrame({'colunas' : train.columns,
                    'tipo': train.dtypes,
                    'missing' : train.isna().sum(),
                    'size' : train.shape[0],
                    'unicos' : train.nunique()})

cons_train_inicial['percentual'] =  round(cons_train_inicial['missing'] / cons_train_inicial['size'],2)


#trata dados que contem Null - o que faz para train, faz para test
train = train.fillna(0)
test = test.fillna(0)


#todos os dados que não são númericos, são tratados aqui
test = pd.concat([pd.get_dummies(test, columns=['TP_SEXO','TP_DEPENDENCIA_ADM_ESC','Q001','Q002','Q006','Q024','Q025','Q047','CO_PROVA_CH','CO_PROVA_LC','CO_PROVA_MT','CO_PROVA_CN'])],axis=1)
train = pd.concat([pd.get_dummies(train, columns=['TP_SEXO','TP_DEPENDENCIA_ADM_ESC','Q001','Q002','Q006','Q024','Q025','Q047','CO_PROVA_CH','CO_PROVA_LC','CO_PROVA_MT','CO_PROVA_CN'])],axis=1)

#existem duas colunas a mais geradas no campo de treinamento:
#CO_PROVA_CH_d5f6d17523d2cce3e4dc0a7f0582a85cec1c15ee 
#CO_PROVA_CN_a27a1efea095c8a973496f0b57a24ac6775d95b0
#como no teste não possui estes valores, acho que podemos retira-las...
train.drop(columns=['CO_PROVA_CH_d5f6d17523d2cce3e4dc0a7f0582a85cec1c15ee', 'CO_PROVA_CN_a27a1efea095c8a973496f0b57a24ac6775d95b0'], axis=1,inplace=True)

#modelo utilizado - ok
rf = RandomForestRegressor(n_estimators=10000,random_state = 42)

#treina o modelo
rf.fit(train, Y)
print (rf.score(train, Y))

#prevendo os resultados
pred = rf.predict(test)

#exportando o resultado
predictions = pd.DataFrame(pred)
final['NU_NOTA_MT'] = predictions
final.to_csv('answer.csv')



# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:28:31 2020

@author: Armstrong
"""

import pandas as pd
import numpy as np

#carrega dados
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")


#isna test
isna_test = test.isna().sum()


#isna train
isna_train = train.isna().sum()

#infotest
#print (test.info())

#infotrain
#print (train.info())

#tipos de dados train
tipos_de_dados_test = test.dtypes

#tipos de dados testes
tipos_de_dados_train = train.dtypes






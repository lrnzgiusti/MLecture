#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:07:58 2017

@author: ince

TODO: discretizzare la variabile FARE intorno un range
"""


import time
cycle_time = time.time() #inizio a contare il tempo d' esecuzione
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Process

df = pd.read_csv('train.csv') #leggo il file
df = df.drop(df.columns[[0, 3, 6, 7, 8, 11]], axis=1) # rimuovo le colonne che ritengo inutili
df = df[df['Age'].notnull()] # rimuovo i record con valore nullo per l' età
df.reset_index(drop=True, inplace=True) #resetto l' indice del DataFrame per velocizzare il calcolo
one_hot = pd.get_dummies(df['Pclass']) # invece di esserci classi {1,2,3} ho un booleano che mi indica l' appartenenza o meno ad una determinata classe
df = df.drop('Pclass', axis=1) #cancello l' attributo classe
df = df.join(one_hot) #inserisco le colonne create in precedenza

df_iter = list(df.itertuples()) #definisco un iteratore, ho uno speedup notevole 

l = np.array([[0]*len(df),[0]*len(df), [0]*len(df)], np.int8) #rinominare l' array, questo array viene utilizzato per discretizzare i valori


""" 1 SE HO LA CABINA 0 ALTRIMENTI """
def discretizeCabine(data):
    for i in range(len(l[0])):
        if(pd.isnull(df_iter[i][5])):
            l[0][i] = 0
        else:
            l[0][i] = 1
          
""" 1 SE MASCHIO 0 ALTRIMENTI """
def discretizeSex(data):
    for i in range(len(l[0])):
         if(df_iter[i][2] == 'male'):
            l[1][i] = 1
         else:
            l[1][i] = 0
      
""" DISCRETIZZO L' ETA' IN BASE AD UN RANGE [0-20, 20-40, 40-60, 60-80] -> {1,2,3,4} """
def discretizeAge(data):
    for i in range(len(l[0])):
        if(df_iter[i][3] > 0 and df_iter[i][3] < 20):
            l[2][i] = 1
        if(df_iter[i][3] >= 20 and df_iter[i][3] < 40):
            l[2][i] = 2
        if(df_iter[i][3] >= 40 and df_iter[i][3] < 60):
            l[2][i] = 3
        if(df_iter[i][3] >= 60 and df_iter[i][3] < 80):
            l[2][i] = 4

            
#        
#for i in range(len(l[0])):
#    if(pd.isnull(df_iter[i][5])):
#        l[0][i] = 0
#    else:
#        l[0][i] = 1
#    if(df_iter[i][2] == 'male'):
#        l[1][i] = 1
#    else:
#        l[1][i] = 0
    
        
""" ESEGUE LE FUNZIONI COME PARAMETRO IN PARALLELO """
def runInParallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()
    
runInParallel(discretizeCabine(df_iter), discretizeSex(df_iter), discretizeAge(df_iter)) #call alla funzione che fa partire i processi in parallelo


scaler = MinMaxScaler(feature_range=(0,50))

vector = np.array(df['Fare'], dtype=np.float16)
# merge
df['Cabin'] = l[0]
df['Sex'] = l[1]
df['Age'] = l[2]
df['Fare'] = scaler.fit_transform(vector.reshape(-1,1))


# data, label split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
label_train = df_train['Survived']
label_test = df_test['Survived']
df_train = df_train.drop('Survived', axis=1)
df_test = df_test.drop('Survived', axis=1)

# resetto gli indici per un calcolo più rapido
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
label_train.reset_index(drop=True, inplace=True)
label_test.reset_index(drop=True, inplace=True)

# funzione che fitta un modello, la definisco per lanciarla in parallelo
def fitModel(model, data, label):
    model.fit(data,label)

# classificatori
for_cla = RandomForestClassifier(n_estimators=500, n_jobs=-1)
nn = MLPClassifier(hidden_layer_sizes=(20,30,4), max_iter=1500, activation='logistic')
sv = SVC(C=2.4, degree=4)
dt = DecisionTreeClassifier()
gnb = GaussianNB()
mnb = MultinomialNB(alpha=1.2)
ada = AdaBoostClassifier(n_estimators=500)
log = LogisticRegression(penalty='l2', max_iter=6000)

#eseguo i fit
runInParallel(fitModel(for_cla, df_train, label_train), 
              fitModel(nn, df_train, label_train), 
              fitModel(sv, df_train, label_train), 
              fitModel(dt, df_train, label_train), 
              fitModel(gnb, df_train, label_train),
              fitModel(mnb, df_train, label_train),
              fitModel(ada, df_train, label_train),
              fitModel(log, df_train, label_train))

# stampo i risultati
print("Forest_Regressor:", for_cla.score(df_test, label_test))
print("NN: ", nn.score(df_test, label_test))
print("SVM: ", sv.score(df_test, label_test))
print("DecisionTree: ", dt.score(df_test, label_test))
print("GNaive Bayes: ", gnb.score(df_test, label_test))
print("MNaive Bayes:", mnb.score(df_test, label_test))
print("Ada Boost: ", ada.score(df_test, label_test))
print("Logistic Regression: ", log.score(df_test, label_test))


print("ended cycle with success in %s seconds" % (time.time()-cycle_time))
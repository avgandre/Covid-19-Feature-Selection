# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:48:01 2020

@author: andre-goncalves
@version:
    Normalização dos dados
    Implementação da Feature Selection Recursiva com Permutation Importance
"""
import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
import numpy as np
import time
from random import sample

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from eli5.sklearn import PermutationImportance
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#Faz a leitura da base
#df = pd.read_csv("Dados/novo_covid_ajustado_com_sintomas1.csv")
df = pd.read_csv("Dados/novo_covid_ajustado_com_sintomas2.csv")

df['RESULTADO'].value_counts()

#Separa a base que está aguardando resultado para predição
dfFuturo = df.query('RESULTADO == 2')
dfFuturo.shape

#Deixa na base de processamento apenas os registros com resultado
df = df.query('RESULTADO == 0 or RESULTADO == 1')
df.shape

#Definindo x, y
features = df.columns.difference(['RESULTADO'])
x = df[features]
y = df['RESULTADO']

#Normalização da base
normalizador = MinMaxScaler(feature_range=(0, 1))
x = pd.DataFrame(normalizador.fit_transform(x))
x.columns = features

#Array do processamento de cada execução
arrayProcessamento = []

sorteados = sample(range(0, 10000), 100)
                   
for rs in sorteados:
    #Tempo de processamento
    tempoInicial = time.time()
    
    #Definição do ramdom_state
    ramdomState=rs
    
    #Cria um dicionário para coletar as importances das features
    featureDictionary = {feature : 0 for feature in features}
    
    #Separa base treinamento e teste
    xTreino, xTeste, yTreino, yTeste = train_test_split(x, y, train_size=0.7, stratify=y, shuffle=True, random_state=ramdomState)
    
    #Balanceamento
    treino = xTreino.join(yTreino)
    qtdeDescartados = treino['RESULTADO'].value_counts()[0]
    qtdeConfirmados = treino['RESULTADO'].value_counts()[1]
    dfDescartados = treino[treino['RESULTADO'] == 0] #Separa a base de descartados
    dfConfirmados = treino[treino['RESULTADO'] == 1] #Separa a base de confirmados
    
    #Under sampling
    print('\nUnder Sampling')
    dfDescartadosUnder = dfDescartados.sample(qtdeConfirmados)
    dfUnder = pd.concat([dfDescartadosUnder, dfConfirmados], axis=0)
    xTreino = dfUnder[features].values
    yTreino = dfUnder['RESULTADO'].values
    
    #Over sampling
    #print('\nOver Sampling')
    #dfConfirmadosOver = dfConfirmados.sample(qtdeDescartados, replace=True, random_state=ramdomState)
    #dfOver = pd.concat([dfDescartados, dfConfirmadosOver], axis=0)
    #xTreino = dfOver[features].values
    #yTreino = dfOver['RESULTADO'].values
    
    #Random Over sampling
    #print('\nRandom Over Sampling')
    #ros = RandomOverSampler(random_state=ramdomState)
    #xTreino, yTreino = ros.fit_resample(treino[features], treino['RESULTADO'])
    #xTreino = xTreino.values
    #yTreino = yTreino.values
    
    #Smote sampling
    #print('\nSmote Sampling')
    #smote = SMOTE(random_state=ramdomState)
    #xTreino, yTreino = smote.fit_resample(treino[features], treino['RESULTADO'])
    #xTreino = xTreino.values
    #yTreino = yTreino.values
    
    #Borderline Smote sampling
    #print('\nBorderline Smote Sampling')
    #borderlineSmote = BorderlineSMOTE(random_state=ramdomState)
    #xTreino, yTreino = borderlineSmote.fit_resample(treino[features], treino['RESULTADO'])
    #xTreino = xTreino.values
    #yTreino = yTreino.values
    
    #Define o classificador
    classifier = RandomForestClassifier(class_weight="balanced", random_state=ramdomState)
    
    #Define o scoring
    score = 'accuracy'
    #score = 'precision'
    #score = 'recall'
    
    #Validação cruzada com embaralhamento
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=ramdomState)
    
    #K-fold
    print('\n========== TUNING PARAMETERS ==========')
    arrayYReal = []
    arrayYPrediction = []
    arrayAcuracia = []
    arrayConfusion = np.array([[0, 0], [0, 0]])
    
    #Grid Search
    paramGrid = {
            'estimator__criterion': ['entropy', 'gini'],
            'estimator__n_estimators': np.arange(3, 101, 1).tolist(),
            'estimator__max_depth':  [None] + np.arange(1, 6, 2).tolist(),
            'estimator__min_samples_split': np.arange(2, 6, 1).tolist(),
            'estimator__min_samples_leaf': np.arange(1, 6, 1).tolist(),
            'estimator__min_weight_fraction_leaf': np.arange(0, 0.6, 0.1).tolist(),
            'estimator__max_features': ['auto'] + np.arange(0.1, 0.6, 0.1).tolist(),
            'estimator__bootstrap': [False, True],
            }
    
    #Permutation Importance
    pi = PermutationImportance(estimator=classifier, scoring=score, cv=kfold, random_state=ramdomState)
    
    #Faz o processamento de treinamento com Tuning e Permutation Importance
    #gridSearch = GridSearchCV(estimator=pi, param_grid=paramGrid, cv=kfold, scoring=score, n_jobs=-1)
    gridSearch = RandomizedSearchCV(estimator=pi, param_distributions=paramGrid, cv=kfold, scoring=score, n_iter=10, n_jobs=-1)
    gridSearch.fit(xTreino, yTreino)
    
    melhorEstimator = gridSearch.best_estimator_
    
    print('\nClassificador:', classifier.__class__)
    print('Score:', score)
    print('\nMelhor parametrização: %s' % gridSearch.best_params_)
    print('Melhor pontuação: %.2f' % gridSearch.best_score_)
    
    #Exibe as 20 melhores features
    #for i in melhorEstimator.feature_importances_.argsort()[::-1][:20]:
    #    print('%s: %.30f +/- %.5f' % (features[i], melhorEstimator.feature_importances_[i], melhorEstimator.feature_importances_std_[i]))   
    
    print('\n========== SELECT FROM MODEL ==========')
    #Feature Selection nas mesmas condições de classificador e folders
    selector = SelectFromModel(melhorEstimator, threshold=(1e-32))
    selector = selector.fit(xTreino, yTreino)
                
    #Faz o corte no dataset considerando apenas as features importantes
    indFeatures = np.where(selector.get_support() == True)[0]
    xTreino = xTreino[:, indFeatures]
    xTeste = xTeste[xTeste.columns[indFeatures]]
    
    print('Qtde features selecionadas: ', len(np.where(selector.get_support() == True)[0]))
    
    #Exibe as features selectionadas
    featureImportances = {}
    for i in selector.estimator.feature_importances_[indFeatures].argsort()[::-1]:
        featureImportances[features[indFeatures[i]]] = selector.estimator.feature_importances_[indFeatures[i]]
        print('%s: %.30f +/- %.5f' % (features[indFeatures[i]],
                                       selector.estimator.feature_importances_[indFeatures[i]],
                                       selector.estimator.feature_importances_std_[indFeatures[i]]))
    
    #Armazena o resultado final das importances das features
    featureDictionary.update(featureImportances)
           
    #K-fold
    print('\n========== VALIDAÇÃO ==========')
    classifier = melhorEstimator.estimator
    
    arrayYReal = []
    arrayYPrediction = []
    arrayAcuracia = []
    arrayConfusion = np.array([[0, 0], [0, 0]])
    
    cv_iter = kfold.split(xTreino, yTreino)
    for treino, teste in cv_iter:
        #Etapa de treinamento
        classifier.fit(xTreino[treino], yTreino[treino])
    
        #Etapa de predição
        yPrediction = classifier.predict(xTreino[teste])
    
        arrayYReal = np.append(arrayYReal, yTreino[teste])
        arrayYPrediction = np.append(arrayYPrediction, yPrediction)
    
        arrayConfusion += confusion_matrix(yTreino[teste], yPrediction, labels=[0, 1])
        arrayAcuracia.append(accuracy_score(yTreino[teste], yPrediction))
    
    print(pd.DataFrame(arrayConfusion, index=['real:descartado', 'real:confirmado'],
                       columns=['pred:descartado', 'pred:confirmado']))
    
    print("\n(TN, FP, FN, TP): %s \n" % arrayConfusion.ravel())
    print(classification_report(arrayYReal, arrayYPrediction, labels=[0, 1])) 
    
    #Teste
    print('\n========== TESTE ==========')
    #Etapa de treinamento
    classifier.fit(xTreino, yTreino)
    
    #Etapa de predição
    yPrediction = classifier.predict(xTeste)
    
    cm = confusion_matrix(yTeste, yPrediction, labels=[0, 1])
    
    print(pd.DataFrame(cm, index=['real:descartado', 'real:confirmado'],
                       columns=['pred:descartado', 'pred:confirmado']))
    
    print("\n(TN, FP, FN, TP): %s \n" % cm.ravel())
    
    print(classification_report(yTeste, yPrediction, labels=[0, 1]))
    
    #Armazena o resultado da etapa de treinamento
    featureDictionary['ResultadoTreinamento'] = arrayConfusion.ravel().tolist()
    
    #Armazena o resultado final da etapa de teste
    featureDictionary['ResultadoTeste'] = cm.ravel().tolist()
    featureDictionary['TempoExecução'] = ((time.time() - tempoInicial) / 60)
    
    arrayProcessamento.append(featureDictionary)

#Salva em disco os dados de todds execução 
dfProcessamento = pd.DataFrame(arrayProcessamento)
dfProcessamento.to_csv('arrayProcessamento.csv', sep=';')
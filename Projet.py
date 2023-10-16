# -*- coding: utf-8 -*-
from prepdata import *
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
# nom des datasets
all_datasets_name = ["abalone8","abalone17","abalone20","autompg",
                 "australian","balance","bankmarketing","bupa","german","glass",
                 "hayes","heart","iono","libras",'newthyroid',"pageblocks","pima","satimage","sonar",
                 "spambase","splice","vehicle","wdbc","wine",'wine4',"yeast3","yeast6"]

# dictionnaire de stockage des resultats
stock_resultat = {}
# liste contenant tous les datasets
dfs = {}
for i in all_datasets_name:
    dfs[i]=data_recovery(i)
    stock_resultat[i] = {}
    #repartition de y
    stock_resultat[i]["repartition_y"] =  dfs[i][1].mean()
    
    # SVM linear
    
    # Algorithme 2



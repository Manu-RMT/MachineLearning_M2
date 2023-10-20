# -*- coding: utf-8 -*-
from prepdata import *
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from ModuleFonction import *
# nom des datasets
# all_datasets_name = ["abalone8","abalone17","abalone20","autompg",
#                   "australian","balance","bankmarketing","bupa","german","glass",
#                   "hayes","heart","iono","libras",'newthyroid',"pageblocks","pima","satimage","sonar",
#                    "spambase","splice","vehicle","wdbc","wine",'wine4',"yeast3","yeast6"]

# all_datasets_name = ["abalone8","abalone17","abalone20","autompg",
#                   "balance","bupa","german","glass",
#                   "hayes","heart","iono","libras",'newthyroid',"pageblocks","pima","satimage","sonar",
#                   "spambase","splice","vehicle","wdbc","wine",'wine4',"yeast3","yeast6"]

# "bankmarketing","australian", "pageblocks", 
all_datasets_name = ["abalone8"]
# dictionnaire de stockage des resultats
stock_resultat = {}
# liste contenant tous les datasets
dfs = {}
for name in all_datasets_name:
    dfs[name]=data_recovery(name)
    stock_resultat[name] = {}
    # repartition de y
    stock_resultat[name]["repartition_y"] =  dfs[name][1].mean()
    # dataset équilibré ?
    if dfs[name][1].mean() < 0.2 or dfs[name][1].mean() > 0.8 :
        stock_resultat[name]["équilibré"] = False
    else : 
        stock_resultat[name]["équilibré"] = True
    
    # sous
    
    # SVM linear
    #stock_resultat[name]['SVM linear'] =  SVM_Test(dfs[name],name)
    # Algorithme 2



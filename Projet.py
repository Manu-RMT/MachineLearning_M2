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
all_datasets_name = ["australian"]
# dictionnaire de stockage des resultats
stock_resultat = {}
# liste contenant tous les datasets
dfs = {}
for name in all_datasets_name:
    dfs[name]=data_recovery(name)
    stock_resultat[name] = {}
    # repartition de y
    stock_resultat[name]["repartition_y"] =  dfs[name][1].mean()
    #
    stock_resultat[name]["major_mino"] =  (dfs[name][1].mean() * len(dfs[name][1])) /  ((1-dfs[name][1].mean()) * len(dfs[name][1]))
   
    
    
    if dfs[name][1].mean() < 0.2 or dfs[name][1].mean() > 0.8 :
        stock_resultat[name]["équilibré"] = False # désiquilibré
        
    else : 
        stock_resultat[name]["équilibré"] = True
    
    # x_equilibre, y_equilibre = reequilibrage
    #  dfs["australian"]= ( dfs["australian"][0], dfs["australian"][1], [0.5]) 
   
    #dfs["australian"]=(['0'])
    # sous
    
    # SVM linear
    stock_resultat[name]['SVM linear'] =  SVM_Test(dfs[name],name)
    # Algorithme 2



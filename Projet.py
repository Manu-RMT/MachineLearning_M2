 # -*- coding: utf-8 -*-
from prepdata import *
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from ModuleFonction import *

patch_sklearn()
ConvergenceWarning('ignore')

# nom des datasets
all_datasets_name = ["abalone8","abalone17","abalone20","autompg",
                  "australian","balance","bankmarketing","bupa","german","glass",
                  "hayes","heart","iono","libras",'newthyroid',"pageblocks","pima","satimage","sonar",
                    "spambase","splice","vehicle","wdbc","wine",'wine4',"yeast3","yeast6"]

# all_datasets_name = ["abalone8","abalone17","abalone20","autompg",
#                   "balance","bupa","german","glass",
#                   "hayes","heart","iono","libras",'newthyroid',"pageblocks","pima","satimage","sonar",
#                   "spambase","splice","vehicle","wdbc","wine",'wine4',"yeast3","yeast6"]

# "bankmarketing","australian", "pageblocks", 
# all_datasets_name = ["abalone8","abalone17"]
# dictionnaire de stockage des resultats
stock_resultat = {}

# liste contenant tous les datasets
dfs = {}
# datasets équilibré
dfs_equilibre = {}
# datasets désiquilibré
dfs_désequilibré = {}

for name in all_datasets_name:
    dfs[name]=data_recovery(name)
    stock_resultat[name] = {}
    # repartition de y
    repartition_y = dfs[name][1].mean()
    stock_resultat[name]["repartition_y"] =  repartition_y
    # major sur minor
    major_minor = (dfs[name][1].mean() * len(dfs[name][1])) /  ((1-repartition_y) * len(dfs[name][1]))
    stock_resultat[name]["major_mino"] =  major_minor
    
       
    # datasets déséquilibré et équilibré&dxcx
    if repartition_y < 0.2 or repartition_y > 0.8 :
        stock_resultat[name]["équilibré"] = False # tableau des résultat désiquilibré
        if(repartition_y < 0.2) :
            type_equilibrage = "sur_echanti"
        else :
            type_equilibrage = 'sous_echanti'
        dfs[name] = reequilibrage(dfs[name], type_equilibrage, major_minor)
        stock_resultat[name]['type equilibrage'] = type_equilibrage  
        
    else : 
        stock_resultat[name]["équilibré"] = True  # tableau des résultat équilibré
        stock_resultat[name]['type equilibrage'] = ""   
    
    # SVM poly sur tous les datasets
    stock_resultat[name]['SVM poly'] =  SVM(dfs[name],name,"poly")
    
    # SVM gauss sur tous les datasets
    stock_resultat[name]['SVM gauss'] =  SVM(dfs[name],name,"rbf")
    
    # SVM linear sur tous les datasets
    stock_resultat[name]['SVM linear'] =  SVM(dfs[name],name,"linear")
        
    # knn 
    stock_resultat[name]['knn'] = Knn(dfs[name], name)
    
    # Arbre de décisions
    stock_resultat[name]['forets aleatoires'] = Arbre_de_decision(dfs[name], name)
    
    # Adaboost
    stock_resultat[name]['Adaboost'] = Adaboost(dfs[name], name)
    
    # Gradient Boosting 
    stock_resultat[name]['Gradient Boosting'] = GradientBoosting(dfs[name], name)


affichage_resultat(stock_resultat)
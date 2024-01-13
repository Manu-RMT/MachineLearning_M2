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
all_datasets_name = ["abalone8","abalone17","abalone20","autompg"]
# all_datasets_name = ["autompg"]
# all_datasets_name = ["abalone8"]
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
    
    temps_traitement = []
    
    # SVM linear sur tous les datasets
    f_mesure,std_f_mesure,score_accuracy,temps_algo = SVM(dfs[name],name,"linear")
    stock_resultat[name]['SVM linear'] = f_mesure
    stock_resultat[name]['SVM linear std'] = std_f_mesure
    stock_resultat[name]['SVM linear accuracy'] = score_accuracy
    temps_traitement.append(temps_algo)
   
    # SVM poly sur tous les datasets
    f_mesure,std_f_mesure,score_accuracy,temps_algo = SVM(dfs[name],name,"poly")
    stock_resultat[name]['SVM poly'] = f_mesure
    stock_resultat[name]['SVM poly std'] = std_f_mesure
    stock_resultat[name]['SVM poly accuracy'] = score_accuracy
    temps_traitement.append(temps_algo)
    
    # SVM gauss sur tous les datasets
    f_mesure,std_f_mesure,score_accuracy,temps_algo = SVM(dfs[name],name,"rbf")
    stock_resultat[name]['SVM gauss'] = f_mesure
    stock_resultat[name]['SVM gauss std'] = std_f_mesure
    stock_resultat[name]['SVM gauss accuracy'] = score_accuracy
    temps_traitement.append(temps_algo)
        
    # knn 
    f_mesure,std_f_mesure,score_accuracy,temps_algo = Knn(dfs[name], name)
    stock_resultat[name]['knn'] = f_mesure
    stock_resultat[name]['knn std'] = std_f_mesure
    stock_resultat[name]['knn accuracy'] = score_accuracy
    temps_traitement.append(temps_algo)
    
    # Arbre de décisions
    f_mesure,std_f_mesure,score_accuracy,temps_algo = Arbre_de_decision(dfs[name], name)
    stock_resultat[name]['forets aleatoires'] = f_mesure
    stock_resultat[name]['forets aleatoires std'] = std_f_mesure
    stock_resultat[name]['forets aleatoires accuracy'] = score_accuracy
    temps_traitement.append(temps_algo)
    
    # Adaboost
    f_mesure,std_f_mesure,score_accuracy,temps_algo = Adaboost(dfs[name], name)
    stock_resultat[name]['Adaboost'] = f_mesure
    stock_resultat[name]['Adaboost std'] = std_f_mesure
    stock_resultat[name]['Adaboost accuracy'] = score_accuracy
    temps_traitement.append(temps_algo)
    
    # Gradient Boosting 
    f_mesure,std_f_mesure,score_accuracy,temps_algo =  GradientBoosting(dfs[name], name)
    stock_resultat[name]['Gradient Boosting'] = f_mesure
    stock_resultat[name]['Gradient Boosting std'] = std_f_mesure
    stock_resultat[name]['Gradient Boosting accuracy'] = score_accuracy
    temps_traitement.append(temps_algo)
    
    stock_resultat[name]['Temps global'] = np.mean(temps_traitement)
    
resultat_final = affichage_resultat(stock_resultat)
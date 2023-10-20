# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:35:36 2023

@author: mramanitra
"""

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np


# Si numérique => on fait moyenne 
# Si alphabétique => rempalcer par la plus grande occurence
def remplace_value_biaise_nan(data, type_data): 
    if type_data == "quanti" :
        str_imputer = SimpleImputer(missing_values=np.nan, strategy= "mean")
    else :
        str_imputer = SimpleImputer(missing_values=np.nan, strategy= "most_frequent")
    str_imputer.fit(data)
    data = str_imputer.transform(data)
    return data


def SVM_Test(df_data,datasets_name,kernel_type="linear"):
    score_per_datasets = {}
    
    # split les données 30% en test et 70% en entrainement
    x_data,y_data = df_data  
    x_train, x_test, y_train, y_test  = train_test_split(x_data,y_data, test_size=0.3,stratify=y_data,shuffle=True)
    
    #k fold (k ici = 5)
    score_moyen_fold_par_hyperparam = []
    # faut donner une grille hyperparametre
    hyper_param = [1,2,4,6,10,12,16]
    skf = StratifiedKFold(n_splits= 5, random_state=None, shuffle=True)
    
    for hp in hyper_param:
        moyenne_k_fold = []
        for i, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
            # print(f"Fold {i}:")
            # print(f"  Train: index={train_index}")
            # print(f"  Test:  index={test_index}")
            x_learn = x_train[train_index]
            y_learn = y_train[train_index]
            x_valid = x_train[test_index]
            y_valid = y_train[test_index]
            
            # test modèle avec le meilleur hyperparamètre 
            clf = svm.SVC(C=hp,kernel=kernel_type)
            clf.fit(x_learn,y_learn)
            # phase de prediction 
            y_predict = clf.predict(x_valid)
            # moyenne de chaque fold
            moyenne_k_fold.append(accuracy_score(y_valid, y_predict))
        # Moyenne de chaque fold par hyperparamètre
        score_moyen_fold_par_hyperparam.append(sum(moyenne_k_fold)/len(moyenne_k_fold))    

    # Meilleur Hyper Paramètre
    meilleur_moyenne = max(score_moyen_fold_par_hyperparam)
    index_meilleur_moyenne_hyperparam = score_moyen_fold_par_hyperparam.index(max(score_moyen_fold_par_hyperparam))
    value_hyper_param = hyper_param[index_meilleur_moyenne_hyperparam]
    print (f"Meilleur Hyper paramètre pour {datasets_name} : {value_hyper_param} ")
    return value_hyper_param
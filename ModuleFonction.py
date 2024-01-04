# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:35:36 2023

@author: mramanitra
"""

from numpy import mean
from numpy import std
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearnex import patch_sklearn 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE

patch_sklearn()
ConvergenceWarning('ignore')

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


    
def reequilibrage(df_data, type_equilibrage,major_minor): 
    x_data,y_data = df_data
    
    # sampling_strategy = faut que ce soit égale à un 0.9 * major_minor => permet de diminier le désequilibrage de 10%
    if type_equilibrage == "sur_echanti":
        # sur echantillonage SMOTE ( ajout de données minoritaires)
        sm = SMOTE() # pour 100 data majoritaire on aura 25 data minoritaires
        x_df,y_df = sm.fit_resample(x_data, y_data)
    else:    
        # sous echantillonage (suppression de données majoritaires)
        rus = RandomUnderSampler() # pour 100 data majoritaire on aura 25 data minoritaires
        x_df,y_df = rus.fit_resample(x_data, y_data)
    
    return x_df,y_df

def SVM_Linear(df_data,datasets_name,kernel_type="linear",avec_equilibrage=False):   
    score_per_datasets = {}
    
    # split les données 30% en test et 70% en entrainement
    x_data,y_data = df_data  
    x_train, x_test, y_train, y_test  = train_test_split(x_data,y_data, test_size=0.3,stratify=y_data,shuffle=True)
    
    #k fold (k ici = 5)
    score_moyen_fold_par_hyperparam = []
    # faut donner une grille hyperparametre
    hyper_param = [1,2,4,6,10,12,16,18,20]
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
            
            # TROP LONG
            # # test modèle avec le meilleur hyperparamètre 
            # clf = svm.SVC(C=hp,kernel=kernel_type)
            # clf.fit(x_learn,y_learn)
            # # phase de prediction 
            # y_predict = clf.predict(x_valid)    
            
            clf = LinearSVC(C=hp,tol=1e-4, max_iter=100)
            clf.fit(x_learn,y_learn)
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


def Adaboost(df_data, datasets_name):
    accu_train_ada = []
    accu_test_ada = []
    score_moyen_adaboost={}
    
    for i in range(1,10):
        x_data,y_data = df_data
        x_train, x_test, y_train, y_test  = train_test_split(x_data,y_data, test_size=0.3,stratify=y_data,shuffle=True)
        # ADABOOST
        #train
        clf = AdaBoostClassifier(n_estimators=100).fit(x_train, y_train)
        y_pred_train_ada = clf.predict(x_train)
        accu_train_ada.append(accuracy_score(y_train, y_pred_train_ada))
        
        #test
        clf = AdaBoostClassifier(n_estimators=100).fit(x_test, y_test)
        y_pred_test_ada = clf.predict(x_test)
        accu_test_ada.append(accuracy_score(y_test, y_pred_test_ada))
    
    score_moyen_adaboost['train'] = (sum(accu_train_ada)/len(accu_train_ada))
    score_moyen_adaboost['test'] = (sum(accu_test_ada)/len(accu_test_ada))  
    print (f"Score Adaboost pour {datasets_name} : {score_moyen_adaboost['test']} ")
    
    return score_moyen_adaboost['test']


def GradientBoosting(df_data, datasets_name):
    x_data,y_data = df_data
    x_train, x_test, y_train, y_test  = train_test_split(x_data,y_data, test_size=0.3,stratify=y_data,shuffle=True)
    
    # define the model
    model = GradientBoostingClassifier()
   # define the evaluation method
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model on the dataset
    n_scores = cross_val_score(model, x_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance (moyenne et écart type)
    print('Score GradientBoosting pour %s : %.3f (%.3f)' % (datasets_name,mean(n_scores), std(n_scores)))
    
    return mean(n_scores)

def Arbre_de_decision(df_data, datasets_name):
    x_data,y_data = df_data
    x_train, x_test, y_train, y_test  = train_test_split(x_data,y_data, test_size=0.3,stratify=y_data,shuffle=True)
    
    # define the model
    model = DecisionTreeClassifier()
    # define the evaluation method
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model on the dataset
    n_scores = cross_val_score(model, x_test, y_test, scoring='roc_auc', cv=cv, n_jobs=-1)
    # report performance (moyenne et écart type)
    print('Score arbre_de_decision pour %s : %.3f (%.3f)' % (datasets_name, mean(n_scores), std(n_scores)))
    
    return mean(n_scores)

def Knn(df_data, datasets_name):
    value_knn = []
    for i in range(1,10):
        x_data,y_data = df_data
        x_train, x_test, y_train, y_test  = train_test_split(x_data,y_data, test_size=0.3,stratify=y_data,shuffle=True)
        knn = KNeighborsRegressor(n_neighbors=10)               
        knn.fit(x_train,y_train)
        predictions = knn.predict(x_test)
    
        score = knn.score(x_test, y_test)
        value_knn.append(score)
    mean_score = sum(value_knn) / len(value_knn)
    print('Score knn pour %s : %.3f' % (datasets_name, mean_score))
        
    return mean_score


def affichage_resultat(tab_resultat):
    print("---------------------------------------------------------------------------------------")
    print(" Datasets ----- SVM linear ----- KNN ----- B-tree ---- adaboost ---- gradient boosting ")
    print("---------------------------------------------------------------------------------------")
    
    num_svm = 0
    num_knn = 0
    num_arb = 0
    num_ada = 0 
    num_grad = 0
    for dataset_name, res in tab_resultat.items():
        svm_linear_value = res["SVM linear"]
        num_svm = num_svm + svm_linear_value
        knn_value = res["knn"]
        num_knn = num_knn + knn_value
        arbre_decision_value = res["forets aleatoires"]
        num_arb = num_arb + arbre_decision_value
        adaboost_value = res["Adaboost"]
        num_ada = num_ada + adaboost_value
        gradient_boosting_value = res["Gradient Boosting"]
        num_grad = num_grad + gradient_boosting_value
        print(" %s ---- %f ------ %f --- %f ---- %f ------ %f " %(dataset_name,svm_linear_value,knn_value,arbre_decision_value,adaboost_value,gradient_boosting_value))
    print("---------------------------------------------------------------------------------------")

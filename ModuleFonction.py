# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:35:36 2023

@author: mramanitra
"""

from numpy import mean,std
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
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import time
import warnings

patch_sklearn()
ConvergenceWarning('ignore')
warnings.filterwarnings('ignore')


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


# reequilibrage des datasets    
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

# calcul la F-mesure
def calcul_fmesure(y_test,y_pred):
        #Matrice de confusion
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        #Precision et rappel
        if (tp+fp) == 0 :
            precision = 0 
        else:
          precision = tp/(tp+fp) 
          
        if (tp+fn) == 0 :
            rappel = 0
        else:
            rappel = tp/(tp+fn)
            
        if (precision+rappel) == 0 :
            f_mesure = 0
        else:
            f_mesure = (2*rappel*precision)/(precision+rappel)
       
        #F_mesure avec beta = 1 pour donner autant de poids au rappel et à la précision
        
        return f_mesure
              
        
def SVM(df_data,datasets_name,kernel_type): 
    print(f"Debut traitement SVM {kernel_type} pour {datasets_name}")
 
    score_per_datasets = {}
    f_mesures = []
    temps_algo = 0
    
    # split les données 30% en test et 70% en entrainement
    for i in range(1,11):        
        x_data,y_data = df_data  
        x_train, x_test, y_train, y_test  = train_test_split(x_data,y_data, test_size=0.3,stratify=y_data,shuffle=True)
        score_moyen_fold_par_hyperparam = []
        # faut donner une grille hyperparametre
        hyper_param = [1,2,4,6,10,12,16,18,20]
        skf = StratifiedKFold(n_splits= 5, random_state=None, shuffle=True)
        
        for hp in hyper_param:
            moyenne_k_fold = []
            for i, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
                x_learn = x_train[train_index]
                y_learn = y_train[train_index]
                x_valid = x_train[test_index]
                y_valid = y_train[test_index]
                
                if kernel_type == "linear" :
                    clf = LinearSVC(C=hp, tol=1e-4, dual=False, max_iter=100)
                    clf.fit(x_learn,y_learn)
                    y_predict = clf.predict(x_valid)
                
                else :
                    #TROP LONG
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
           
        if kernel_type == "linear" :
            clf = LinearSVC(C=value_hyper_param,tol=1e-4, max_iter=100)
            clf.fit(x_learn,y_learn)
            start = time.perf_counter()
            y_predict = clf.predict(x_test)
            end = time.perf_counter()
            temps_algo = end - start
        else :
            #TROP LONG
            # test modèle avec le meilleur hyperparamètre 
            clf = svm.SVC(C=value_hyper_param,kernel=kernel_type)
            clf.fit(x_train,y_train)
            # phase de prediction 
            start = time.perf_counter() 
            y_predict = clf.predict(x_test)  
            end = time.perf_counter()
            temps_algo = end - start
            
        f_mesures.append(calcul_fmesure(y_test,y_predict))

    f_mesure = mean(f_mesures)
    std_f_mesure = mean(f_mesures)
    score_accuracy = mean(score_moyen_fold_par_hyperparam)
    
    
    print(f"Fin traitement SVM {kernel_type} pour {datasets_name} avec un temps de {temps_algo}")
    
    return f_mesure,std_f_mesure,score_accuracy,temps_algo



def Adaboost(df_data, datasets_name):
    print(f"Debut traitement Adaboost pour {datasets_name}")
   
    accu_train_ada = []
    accu_test_ada = []
    f_mesures = []
    
    # ADABOOST   
    for i in range(1,11):
        x_data,y_data = df_data
        x_train, x_test, y_train, y_test  = train_test_split(x_data,y_data, test_size=0.3,stratify=y_data,shuffle=True)
         
        #train
        clf = AdaBoostClassifier(n_estimators=100).fit(x_train, y_train)
        y_pred_train_ada = clf.predict(x_train)
        accu_train_ada.append(accuracy_score(y_train, y_pred_train_ada))
        
        #test
        clf = AdaBoostClassifier(n_estimators=100).fit(x_test, y_test)
        start = time.perf_counter() 
        y_pred_test_ada = clf.predict(x_test)
        end = time.perf_counter() 
        accu_test_ada.append(accuracy_score(y_test, y_pred_test_ada))
        
        f_mesures.append(calcul_fmesure(y_test,y_pred_test_ada))
              
    f_mesure = mean(f_mesures)
    std_f_mesure = std(f_mesures)
    score_accuracy = mean(accu_test_ada)
    
    temps_algo = end - start
    print(f"Fin traitement Adaboost pour {datasets_name} avec un temps de {temps_algo}")
    
    
    return f_mesure,std_f_mesure,score_accuracy,temps_algo



def GradientBoosting(df_data, datasets_name):
    print(f"Debut traitement GradientBoosting pour {datasets_name}")
   
    x_data,y_data = df_data
    x_train, x_test, y_train, y_test  = train_test_split(x_data,y_data, test_size=0.3,stratify=y_data,shuffle=True)
    model = GradientBoostingClassifier()
    f_mesures = []
    scores = []
    for i in range(1,11): 
        x_train, x_test, y_train, y_test  = train_test_split(x_data,y_data, test_size=0.3,stratify=y_data,shuffle=True)
        model.fit(x_train,y_train)
        start = time.perf_counter()  
        y_predict = model.predict(x_test)
        end = time.perf_counter() 
        
        f_mesures.append(calcul_fmesure(y_test, y_predict))
        scores.append(accuracy_score(y_test, y_predict))
        
    score_accuracy = mean(scores)
    f_mesure = np.mean(f_mesures)
    std_f_mesure = np.std(f_mesures)

    temps_algo = end - start
    print(f"Fin traitement GradientBoosting pour {datasets_name}  avec un temps de {temps_algo}")

    return f_mesure,std_f_mesure,score_accuracy,temps_algo



def Arbre_de_decision(df_data, datasets_name):
    print(f"Debut traitement Arbre de decision  pour {datasets_name} ")
      
    f_mesures = []
    scores=[]
    for i in range(1,11):
        x_data,y_data = df_data
        x_train, x_test, y_train, y_test  = train_test_split(x_data,y_data, test_size=0.3,stratify=y_data,shuffle=True)
        # define the model
        model = DecisionTreeClassifier()
        model.fit(x_train,y_train)
        start = time.perf_counter()
        y_predict = model.predict(x_test)
        end = time.perf_counter() 
        
        f_mesures.append(calcul_fmesure(y_test, y_predict))    
        scores.append(accuracy_score(y_test, y_predict))
        
    f_mesure = np.mean(f_mesures)
    std_f_mesure = np.std(f_mesures)
    score_accuracy = mean(scores)     
     
    temps_algo = end - start   
    print(f"Fin traitement Arbre de decision  pour {datasets_name}  avec un temps de {temps_algo} ")
    
    return f_mesure,std_f_mesure,score_accuracy,temps_algo


def Knn(df_data, datasets_name):
    print(f"Debut traitement KNN pour {datasets_name}")
       
    accuracy_scores = []
    f_mesures = []
    for i in range(1,11):
        x_data,y_data = df_data
        x_train, x_test, y_train, y_test  = train_test_split(x_data,y_data, test_size=0.3,stratify=y_data,shuffle=True)
        knn = KNeighborsClassifier(n_neighbors=i)               
        knn.fit(x_train,y_train)
        start = time.perf_counter()  
        y_predict = knn.predict(x_test)
        end = time.perf_counter() 
        
        f_mesures.append(calcul_fmesure(y_test, y_predict))
        accuracy_scores.append(knn.score(x_test, y_test))
          
    f_mesure = mean(f_mesures)
    std_f_mesure = np.std(f_mesures)
    score_accuracy = mean(accuracy_scores)
    
    
    temps_algo = end - start
    print(f"Debut traitement KNN pour {datasets_name}  avec un temps de {temps_algo}")        
    return f_mesure,std_f_mesure,score_accuracy,temps_algo



def classement_par_algo(accuracy_values):
    indices = list(range(len(accuracy_values)))
    indices.sort(reverse = True, key=lambda x: accuracy_values[x])
    output = [0] * len(indices)
    for i, x in enumerate(indices):
        output[x] = i
    return output

def remplissage_pied_dataframe(df,data, type_remplissage):
    if type_remplissage == "rank":
        df = df._append({'Datasets': data[0],
                            'SVM linear': data[1],
                            'SVM linear std': np.nan,
                            'SVM poly': data[2],
                            'SVM poly std': np.nan,
                            'SVM gauss': data[3],
                            'SVM gauss std': np.nan,
                            'KNN': data[4],
                            'KNN std': np.nan,
                            'Arbre de decision': data[5],
                            'Arbre de decision std': np.nan,
                            'Adaboost': data[6],
                            'Adaboost std': np.nan,
                            'Gradient boosting': data[7],
                            'Gradient boosting std': np.nan,
                            'Temps Traitement': data[8],
                      }
                      ,ignore_index=True
                     )
    else:
        df = df._append({'Datasets': data[0],
                            'SVM linear': data[1],
                            'SVM linear std':data[2],
                            'SVM poly': data[3],
                            'SVM poly std': data[4],
                            'SVM gauss': data[5],
                            'SVM gauss std': data[6],
                            'KNN': data[7],
                            'KNN std': data[8],
                            'Arbre de decision': data[9],
                            'Arbre de decision std': data[10],
                            'Adaboost': data[11],
                            'Adaboost std': data[12],
                            'Gradient boosting': data[13],
                            'Gradient boosting std': data[14],
                            'Temps Traitement': data[15],
                      }
                      ,ignore_index=True
                     )    
    return df


def affichage_resultat(tab_resultat):
    # entete du dataframe
    df = pd.DataFrame(columns=['Datasets','SVM linear', 'SVM linear std','SVM poly','SVM poly std',
                               'SVM gauss', 'SVM gauss std','KNN', 'KNN std', 'Arbre de decision', 'Arbre de decision std',
                               'Adaboost','Adaboost std','Gradient boosting', 'Gradient boosting std',
                               'Temps Traitement' ]) 
    df.to_csv(r'resultat_apprentissage0.csv',index=False,sep=';')    

    for dataset_name, res in tab_resultat.items():
                
        rapport_maj_min = res["major_mino"] * 100
        percent_maj_min = "{:.2f}%".format(rapport_maj_min)
        Temps_traitement = res["Temps global"]
        svm_linear_value = res["SVM linear"]
        svm_poly_value = res["SVM poly"] 
        svm_gauss_value = res["SVM gauss"]
        knn_value = res["knn"]
        arbre_decision_value = res["arbre de decision"]
        adaboost_value = res["Adaboost"]
        gradient_boosting_value = res["Gradient Boosting"]
        
        svm_linear_value_std = res["SVM linear std"]
        svm_poly_value_std = res["SVM poly std"] 
        svm_gauss_value_std = res["SVM gauss std"]
        knn_value_std = res["knn std"]
        arbre_decision_value_std = res["arbre de decision std"]
        adaboost_value_std = res["Adaboost std"]
        gradient_boosting_value_std = res["Gradient Boosting std"]
        
        
        valeurs_rank = [svm_linear_value,svm_poly_value,svm_gauss_value,knn_value,arbre_decision_value,adaboost_value,gradient_boosting_value]
        Average_Rank = [classement_par_algo(valeurs_rank)]
        
        df = df._append({'Datasets': percent_maj_min + " " + dataset_name,
                        'SVM linear': svm_linear_value,
                        'SVM linear std': svm_linear_value_std,
                        'SVM poly': svm_poly_value,
                        'SVM poly std': svm_poly_value_std,
                        'SVM gauss': svm_gauss_value,
                        'SVM gauss std': svm_gauss_value_std,
                        'KNN': knn_value, 
                        'KNN std': knn_value_std, 
                        'Arbre de decision': arbre_decision_value,
                        'Arbre de decision std': arbre_decision_value_std,
                        'Adaboost': adaboost_value,
                        'Adaboost std': adaboost_value_std ,
                        'Gradient boosting': gradient_boosting_value,
                        'Gradient boosting std': gradient_boosting_value_std,
                        'Temps Traitement': Temps_traitement,
                  }
                  ,ignore_index=True
                 )
        
    df.to_csv(r'resultat_apprentissage1.csv',index=False,sep=';')    
    # moyenne
    mean_svm_linear = df.describe()["SVM linear"]["mean"]
    mean_svm_poly = df.describe()["SVM poly"]["mean"]
    mean_svm_gauss = df.describe()["SVM gauss"]["mean"]
    mean_knn = df.describe()["KNN"]["mean"]
    mean_arb_decision = df.describe()["Arbre de decision"]["mean"]
    mean_adaboost = df.describe()["Adaboost"]["mean"]
    mean_gradient = df.describe()["Gradient boosting"]["mean"]
    mean_temps = df.describe()["Temps Traitement"]["mean"]
    
    # moyenne ecart-type
    mean_svm_linear_std = df.describe()["SVM linear std"]["mean"]
    mean_svm_poly_std = df.describe()["SVM poly std"]["mean"]
    mean_svm_gauss_std = df.describe()["SVM gauss std"]["mean"]
    mean_knn_std = df.describe()["KNN std"]["mean"]
    mean_arb_decision_std = df.describe()["Arbre de decision std"]["mean"]
    mean_adaboost_std = df.describe()["Adaboost std"]["mean"]
    mean_gradient_std = df.describe()["Gradient boosting std"]["mean"]
    
    Average_Rank = [sum(x)/len(x) + 1  for x in zip(*Average_Rank)] # moyenne des classements (+1 pour que les classements commence à 1 plutôt que 0)
    Average_Rank.append(np.nan)
    Average_Rank.insert(0, "Average Rank")
   
    
    # pied du dataframe
    mean_global =["Mean", mean_svm_linear, mean_svm_linear_std, mean_svm_poly, mean_svm_poly_std, 
                          mean_svm_gauss, mean_svm_gauss_std, mean_knn, mean_knn_std,
                          mean_arb_decision, mean_arb_decision_std, mean_adaboost, mean_adaboost_std, 
                          mean_gradient, mean_gradient_std, mean_temps]
    
    
    df= remplissage_pied_dataframe(df, mean_global,"mean")
    df= remplissage_pied_dataframe(df, Average_Rank,"rank")
         
    df.to_csv(r'resultat_apprentissage.csv',index=False,sep=';')    
    df.style.map(color_negative_red) #applymap
  
    return df
  
    
    

def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'blue' if val > 90 else 'black'
    return 'color: % s' % color
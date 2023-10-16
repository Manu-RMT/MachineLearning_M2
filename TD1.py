# -*- coding: utf-8 -*-
from prepdata import *
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
# nom des datasets
all_datasets_name = ["abalone8","abalone17","abalone20","autompg",
                 "australian","balance","bankmarketing","bupa","german","glass",
                 "hayes","heart"]

# liste contenant tous les datasets
dfs = {}
for i in all_datasets_name:
    dfs[i]=data_recovery(i)
    

df_abalone8 = data_recovery("glass")
x_abalone8,y_abalone8 = df_abalone8  
#taille abalone8 
taille_abalone8 = len(y_abalone8)


# split les données 30% en test et 70% en entrainement
x_train, x_test, y_train, y_test  = train_test_split(x_abalone8,y_abalone8, test_size=0.3,stratify=y_abalone8,shuffle=True)

#k fold 
score_moyen_fold_par_hyperparam = []
# faut donner une grille hyperparametre
hyper_param = [1,2,4,6,10,12,16]
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)

for hp in hyper_param:
    moyenne_k_fold = []
    for i, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        x_learn = x_train[train_index]
        y_learn = y_train[train_index]
        x_valid = x_train[test_index]
        y_valid = y_train[test_index]
        
        #  tester le modèle avec le meilleur hyperparamètre sur les données tests
        clf = svm.SVC(C=hp,kernel="linear")
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
print(f"Meilleur Hyper paramètre : {value_hyper_param} ")

accuracy_per_tour.append(value_hyper_param)



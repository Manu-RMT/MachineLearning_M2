from prepdata import *
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from ModuleFonction import *
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

all_datasets_name = ["abalone17"]
dfs = {}
for name in all_datasets_name:
    dfs[name]=data_recovery(name)
    
x_data,y_data = dfs[name]  
x_train, x_test, y_train, y_test  = train_test_split(x_data,y_data, test_size=0.3,stratify=y_data,shuffle=True)

clf_tab = {}
accu_train = []
accu_test = []
accu_train_ada = []
accu_test_ada = []
score_moyen_bagging={}
score_moyen_adaboost={}

for i in range(1,10):
    # BAGGING
    #train
    clf = BaggingClassifier(base_estimator=SVC(), n_estimators=10).fit(x_train, y_train)
    y_pred_train = clf.predict(x_train)
    accu_train.append(accuracy_score(y_train, y_pred_train))
    
    #test
    clf = BaggingClassifier(base_estimator=SVC(), n_estimators=10).fit(x_test, y_test)
    y_pred_test = clf.predict(x_test)
    accu_test.append(accuracy_score(y_test, y_pred_test))
    
    
    # ADABOOST
    #train
    clf = AdaBoostClassifier(n_estimators=100).fit(x_train, y_train)
    y_pred_train_ada = clf.predict(x_train)
    accu_train_ada.append(accuracy_score(y_train, y_pred_train_ada))
    
    #test
    clf = AdaBoostClassifier(n_estimators=100).fit(x_test, y_test)
    y_pred_test_ada = clf.predict(x_test)
    accu_test_ada.append(accuracy_score(y_test, y_pred_test_ada))
   
    
    
score_moyen_bagging['train'] = (sum(accu_train)/len(accu_train))
score_moyen_bagging['test'] = (sum(accu_test)/len(accu_test))

score_moyen_adaboost['train'] = (sum(accu_train_ada)/len(accu_train_ada))
score_moyen_adaboost['test'] = (sum(accu_test_ada)/len(accu_test_ada))

score_methode_ensembliste={}
score_methode_ensembliste['bagging'] = score_moyen_bagging
score_methode_ensembliste['adaboost'] = score_moyen_adaboost

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:09:28 2023

@author: mramanitra
"""

from prepdata import *
from ModuleFonction import *
import numpy as np

dataCSV = loadCSV_Panda("VisaPremier.csv")
data_clean = dataCSV.drop(columns=["cartevp","sexer","matricul"])
data_clean = data_clean.astype({'departem': 'object'})

data_qt = data_clean.select_dtypes(include=['int64','float64'])
data_quali = data_clean.select_dtypes(include=['object'])


#test = remplace_point_nan(data_qt,"quanti")
numpy_qt = remplace_value_biaise_nan(data_qt,"quanti")
numpy_quali = remplace_value_biaise_nan(data_quali,"quali")

# concatenation des valeurs
concatenate_array = np.concatenate((numpy_qt,numpy_quali),axis =1)

column_data_quanti = list(data_qt.columns)
column_data_quali = list(data_quali.columns)
all_column_name = column_data_quanti + column_data_quali


data_ok = pd.DataFrame(concatenate_array, columns= all_column_name)

train = train_test(data_ok)
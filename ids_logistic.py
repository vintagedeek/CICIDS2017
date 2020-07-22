#!/usr/bin/env python3

# Load the top modules that are used in multiple places
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ids_utils import *

def ids_logistic():
	"""
	Classify processed data set stored as csv file using logistic regression
	Print: accuracy, confusion matrix, f1 score on the validation data set
	Input:
		None	
	Returns:
		None
	"""

	df = ids_load_df_from_csv (outdir, file)
	X_train, X_val, X_test, y_train, y_val, y_test = ids_split(df)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	# print ('mean', X_train.mean(axis=0))
	
	# max_iter set to a large value to prevent LogisticRegression() from complaining that
	# it is not coverging
	logreg = LogisticRegression(max_iter=10000)
	logreg.fit(X_train, y_train)

	X_val = scaler.transform(X_val)
	# print ('mean', X_val.mean(axis=0))

	y_pred = logreg.predict(X_val)
	
	ids_metrics(y_val, y_pred)

ids_logistic()

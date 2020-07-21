#!/usr/bin/env python3

# Load the top modules that are used in multiple places
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import ids_utils as iu

def ids_logistic():
	"""
	Classify processed data set stored as csv file using logistic regression
	Print: accuracy, confusion matrix, f1 score on the validation data set
	Input:
		None	
	Returns:
		None
	"""

	df = iu.ids_load_df_from_csv (iu.outdir, iu.file)
	X_train, X_val, X_test, y_train, y_val, y_test = iu.ids_train_test_split(df)

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
	
	print('Accuracy of logistic regression classifier on val set: {:.4f}'.format(logreg.score(X_val, y_val)))
	
	cm = confusion_matrix(y_val, y_pred)
	print(cm)
	
	f1 = f1_score(y_val, y_pred, average='weighted')
	print('F1 score of logistic regression classifier on val set: {:.4f}'.format(f1))

ids_logistic()

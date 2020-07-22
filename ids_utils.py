#!/usr/bin/env python3

"""
Data Cleaning and Utility functions for CICIDS 2017 data
"""

# Load the top modules that are used in multiple places
import numpy as np
import pandas as pd

# Some global variables to drive the script
# The indir should match the location of the data
indir = '~/Tmp/MachineLearningCVE/'
outdir='~/Tmp/MachineLearningCVE/processed/'
file_base='cicids2017.csv'
file='bal-cicids2017.csv'

# The test directory is used to test code on 
# toy data before running it on the full CICIDS 2017 data
# indir = '~/Tmp/MachineLearningCVE/test/'

# Column name mapping from original data to compact form
# All the X** are features and the YY is the label
column_names = {
 ' Destination Port' : 'X1',
 ' Flow Duration' : 'X2', 
 ' Total Fwd Packets' : 'X3', 
 ' Total Backward Packets' : 'X4', 
 'Total Length of Fwd Packets' : 'X5', 
 ' Total Length of Bwd Packets' : 'X6', 
 ' Fwd Packet Length Max' : 'X7', 
 ' Fwd Packet Length Min' : 'X8', 
 ' Fwd Packet Length Mean' : 'X9', 
 ' Fwd Packet Length Std' : 'X10', 
 'Bwd Packet Length Max' : 'X11', 
 ' Bwd Packet Length Min' : 'X12', 
 ' Bwd Packet Length Mean' : 'X13', 
 ' Bwd Packet Length Std' : 'X14', 
 'Flow Bytes/s' : 'X15', 
 ' Flow Packets/s' : 'X16', 
 ' Flow IAT Mean' : 'X17', 
 ' Flow IAT Std' : 'X18', 
 ' Flow IAT Max' : 'X19', 
 ' Flow IAT Min' : 'X20', 
 'Fwd IAT Total' : 'X21', 
 ' Fwd IAT Mean' : 'X22', 
 ' Fwd IAT Std' : 'X23', 
 ' Fwd IAT Max' : 'X24', 
 ' Fwd IAT Min' : 'X25', 
 'Bwd IAT Total' : 'X26', 
 ' Bwd IAT Mean' : 'X27', 
 ' Bwd IAT Std' : 'X28', 
 ' Bwd IAT Max' : 'X29', 
 ' Bwd IAT Min' : 'X30', 
 'Fwd PSH Flags' : 'X31', 
 ' Bwd PSH Flags' : 'X32', 
 ' Fwd URG Flags' : 'X33', 
 ' Bwd URG Flags' : 'X34', 
 ' Fwd Header Length' : 'X35', 
 ' Bwd Header Length' : 'X36', 
 'Fwd Packets/s' : 'X37', 
 ' Bwd Packets/s' : 'X38', 
 ' Min Packet Length' : 'X39', 
 ' Max Packet Length' : 'X40', 
 ' Packet Length Mean' : 'X41', 
 ' Packet Length Std' : 'X42', 
 ' Packet Length Variance' : 'X43', 
 'FIN Flag Count' : 'X44', 
 ' SYN Flag Count' : 'X45', 
 ' RST Flag Count' : 'X46', 
 ' PSH Flag Count' : 'X47', 
 ' ACK Flag Count' : 'X48', 
 ' URG Flag Count' : 'X49', 
 ' CWE Flag Count' : 'X50', 
 ' ECE Flag Count' : 'X51', 
 ' Down/Up Ratio' : 'X52', 
 ' Average Packet Size' : 'X53', 
 ' Avg Fwd Segment Size' : 'X54', 
 ' Avg Bwd Segment Size' : 'X55', 
 ' Fwd Header Length.1' : 'X56', 
 'Fwd Avg Bytes/Bulk' : 'X57', 
 ' Fwd Avg Packets/Bulk' : 'X58', 
 ' Fwd Avg Bulk Rate' : 'X59', 
 ' Bwd Avg Bytes/Bulk' : 'X60', 
 ' Bwd Avg Packets/Bulk' : 'X61', 
 'Bwd Avg Bulk Rate' : 'X62', 
 'Subflow Fwd Packets' : 'X63', 
 ' Subflow Fwd Bytes' : 'X64', 
 ' Subflow Bwd Packets' : 'X65', 
 ' Subflow Bwd Bytes' : 'X66', 
 'Init_Win_bytes_forward' : 'X67', 
 ' Init_Win_bytes_backward' : 'X68', 
 ' act_data_pkt_fwd' : 'X69', 
 ' min_seg_size_forward' : 'X70', 
 'Active Mean' : 'X71', 
 ' Active Std' : 'X72', 
 ' Active Max' : 'X73', 
 ' Active Min' : 'X74', 
 'Idle Mean' : 'X75', 
 ' Idle Std' : 'X76', 
 ' Idle Max' : 'X77', 
 ' Idle Min' : 'X78', 
 ' Label': 'YY'
}

# label names (YY) in the data and their
# mapping to numerical values
label_names = {
 'BENIGN' : 0,
 'FTP-Patator' : 1,
 'SSH-Patator' : 2,
 'DoS slowloris' : 3,
 'DoS Slowhttptest': 4,
 'DoS Hulk' : 5,
 'DoS GoldenEye' : 6,
 'Heartbleed' : 7,
 'Web Attack � Brute Force': 8,
 'Web Attack � XSS': 8,
 'Web Attack � Sql Injection': 8,
 'Infiltration': 9,
 'Bot' : 10,
 'PortScan' : 11,
 'DDoS' : 12,
}

ids_features = 76
ids_classes = 13

def ids_combine(indir, outdir, file):
	"""
	Combine all csv files to produce a single csv file 
	Input:
		indir: directory for set of csv files
		outdir: directory for output csv file
		file: output csv file
	Returns:
		None
	"""

	import os
	import glob
	os.chdir(indir)
	extension = 'csv'
	all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

	# combine all files in the list
	df = pd.concat([pd.read_csv(f) for f in all_filenames ])

	# Drop columns 14 and 15 that have Nan and Infinity in them
	df.rename(columns = column_names, inplace=True)
	df.drop(columns=['X15', 'X16'], inplace=True)

	# Convert string labels to numeric
	df['YY'].replace(label_names, inplace=True)

	# export to csv
	df.to_csv(outdir+file, index=False)

def ids_balance(dir, infile, outfile):
	"""
	Balance dataset using a heuristic
	Input:
		dir: directory for csv file
		infile: input csv file
		outfile: output csv file
	Returns:
		None
	"""

	from sklearn.utils import resample
	n = 8000

	df = pd.read_csv(dir + infile, delimiter=',')
	df0 = df[df.YY == 0]
	df1 = df[df.YY == 1]
	df2 = df[df.YY == 2]
	df3 = df[df.YY == 3]
	df4 = df[df.YY == 4]
	df5 = df[df.YY == 5]
	df6 = df[df.YY == 6]
	df7 = df[df.YY == 7]
	df8 = df[df.YY == 8]
	df9 = df[df.YY == 9]
	df10 = df[df.YY == 10]
	df11 = df[df.YY == 11]
	df12 = df[df.YY == 12]
	
	df0 = resample(df0, replace=False, n_samples=5*n, random_state=123)
	df1 = resample(df1, replace=True, n_samples=n, random_state=123)
	df2 = resample(df2, replace=True, n_samples=n, random_state=123)
	df3 = resample(df3, replace=True, n_samples=n, random_state=123)
	df4 = resample(df4, replace=True, n_samples=n, random_state=123)
	df5 = resample(df5, replace=False, n_samples=n, random_state=123)
	df6 = resample(df6, replace=False, n_samples=n, random_state=123)
	df7 = resample(df7, replace=True, n_samples=n, random_state=123)
	df8 = resample(df8, replace=True, n_samples=n, random_state=123)
	df9 = resample(df9, replace=True, n_samples=n, random_state=123)
	df10 = resample(df10, replace=True, n_samples=n, random_state=123)
	df11 = resample(df11, replace=False, n_samples=n, random_state=123)
	df12 = resample(df12, replace=False, n_samples=n, random_state=123)

	df_sampled = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12])

	print (df_sampled.YY.value_counts())

	# export to csv
	df_sampled.to_csv(dir + outfile, index=False)

def ids_load_df_from_csv(dir, file):
	"""
	Load dataframe from csv file
	Input:
		dir: directory for csv file
		file: csv file
	Returns:
		Pandas dataframe corresponding to processed and saved csv file
	"""

	df = pd.read_csv(dir + file)

	print ("load Dataframe shape", df.shape)

	return df

def ids_split(df):
	"""
	Input:
		Dataframe that has columns of covariates followed by a column of labels
	Returns:
		X_train, X_val, X_test, y_train, y_val, y_test
	"""

	from sklearn.model_selection import train_test_split

	numcols = len(df.columns)
	print("df.shape", df.shape)

	X = df.iloc[:, 0:numcols-1]
	y = df.loc[:, 'YY']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=41)
	print ("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
	print ("X_val.shape", X_val.shape, "y_val.shape", y_val.shape)
	print ("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)

	return X_train, X_val, X_test, y_train, y_val, y_test

def ids_metrics(y_actual, y_pred):
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import confusion_matrix
	from sklearn.metrics import f1_score

	acc = accuracy_score (y_actual, y_pred)
	print('Accuracy of classifier: {:.4f}'.format(acc))

	f1 = f1_score(y_actual, y_pred, average='macro')
	print('F1 score: {:.4f}'.format(f1))

	cm = confusion_matrix (y_actual, y_pred)
	print(cm)

def ids_check_version():
	""" Prints Python version in use """
	import sys
	print (sys.version)


# def main():
ids_check_version()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# uncomment to clean and combine data files
# ids_combine(indir, outdir, file_base)

# uncomment to create a class-balanaced version of the data
# ids_balance (outdir, file_base, file)

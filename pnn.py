import numpy as np
import pandas as pd
import read_data
import os
from tqdm import tqdm

from sklearn.metrics import accuracy_score, \
							confusion_matrix, \
							precision_score, \
							recall_score


def rbf(centre, x, sigma):
	centre = centre.reshape(1, -1)

	temp = -np.sum((centre - x) ** 2, axis = 1)
	temp = temp / (2 * sigma * sigma)
	temp = np.exp(temp)
	gaussian = np.sum(temp)
	
	return gaussian


def subset_by_class(data, labels):
	x_train_subsets = []

	for l in labels:

		#Per ogni riga in data y train, verificare se tale riga è uguale alla l corrente
		#In esito positivo, inserire l'indice della riga in indices

		indices = []
		for j, row in enumerate(data['y_train']):
			if (np.array_equal(row,l)):
				indices.append(j)
		
		# print("\n\nPer la label", l, " gli indici sono:", indices)
		x_train_subsets.append(data['x_train'][indices, :])

	return x_train_subsets


def PNN(data):
	num_test_set = data['x_test'].shape[0]

	#Anche questo va modificato, perchè non è detto che in y_train ci sia almeno un sample di tutte e 256 persone
	# labels = np.unique(data['y_train'], axis=0)

	labels = []

	# Inizializza una matrice vuota per contenere i vettori one-hot-encoded
	one_hot_matrix = np.zeros((256, 256))

	# Genera i vettori one-hot-encoded
	for i in range(256):
		one_hot_matrix[i][i] = 1

	for row in one_hot_matrix:
		labels.append(list(row))
	
	#print(labels)
 
	num_class = len(labels)

	print("Num classes: ",num_class)

	sigma = 10

	x_train_subsets = subset_by_class(data, labels)

	print("Lenght of x_train_subsets: ",len(x_train_subsets))
	for a,subset in enumerate(x_train_subsets):
		print("Subset ", a, "has shape: ",np.shape(subset))

	summation_layer = np.zeros(num_class)
	predictions = np.zeros(num_test_set)

	i = 0

	for test_point in tqdm(data['x_test'], desc="Forecasting", ncols=100):
		#print("Test point",i," shape: ",np.shape(test_point))
		for j, subset in enumerate(x_train_subsets):
			if (np.shape(subset)[0] != 0):
				dim0 = np.shape(subset)[0]
				dim1 = np.shape(subset)[1]
				subset = subset.reshape((1,dim0,dim1))
				#print("\tNEW subset ",j," shape: ",np.shape(subset))
				summation_layer[j] = np.sum(
				rbf(test_point, subset[0], sigma)
			) / subset[0].shape[0]
			else:
				#print("\tSubset shape", j, "shape: ", np.shape(subset))
				summation_layer[j] = 0
			#print("Summation layer ", j, ": ", summation_layer[j])
	
		predictions[i] = np.argmax(summation_layer)
		
		i = i + 1
	
	return predictions



def print_metrics(y_test, predictions):
	print("Y test:", np.shape(y_test))
	print("Predictions: ",np.shape(predictions))

	#print('Confusion Matrix')
	#print(confusion_matrix(y_test, predictions))
	print('Accuracy: {}'.format(accuracy_score(y_test, predictions)))
	#print('Precision: {}'.format(precision_score(y_test, predictions, average = 'micro')))
	#print('Recall: {}'.format(recall_score(y_test, predictions, average = 'micro')))
	

if __name__ == '__main__':

	##########################################################################
	# OGNI METODO DI READ_DATA DEVE ORA ESSERE CHIAMATO DA read_data #
	##########################################################################

	train_csv_filename = "train_dataset.csv"
	test_csv_filename = "test_dataset.csv"

	train_directory = "Dataset/Train"
	test_directory = "Dataset/Test"

	# Il dataset viene generato esclusivamente se non presente nella current working directory
	if not os.path.exists(train_csv_filename) or not os.path.exists(test_csv_filename):
		# Creazione del dataset di training
		read_data.create_csv(train_csv_filename, train_directory)
		# Creazione del dataset di test
		read_data.create_csv(test_csv_filename, test_directory)

	datasets = [train_csv_filename, test_csv_filename]

	data = read_data.create_data_correctly(train_csv_filename, test_csv_filename)

	predictions = PNN(data)

	print(type(predictions))

	pd.DataFrame(predictions).to_csv("predictions.csv")

	print_metrics(data['y_test_before_ohe'], predictions)

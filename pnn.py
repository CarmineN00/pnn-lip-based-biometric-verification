import numpy as np

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
		indices = np.where(data['y_train'] == l)
		x_train_subsets.append(data['x_train'][indices, :])

	return x_train_subsets


def PNN(data):
	num_test_set = data['x_test'].shape[0]

	labels = np.unique(data['y_train'])
	
	num_class = len(labels)

	print("Numero di classi", num_class)
	print("Lunghezza X_TEST", len(data['x_test']))
	print("Lunghezza Y_TEST", len(data['y_test']))
	print("Lunghezza X_TRAIN", len(data['x_train']))
	print("Lunghezza Y_TRAIN", len(data['y_train']))

	sigma = 10

	x_train_subsets = subset_by_class(data, labels)

	summation_layer = np.zeros(num_class)
	predictions = np.zeros(num_test_set)

	i = 0

	for test_point in tqdm(data['x_test'], desc="Forecasting", ncols=100):
		for j, subset in enumerate(x_train_subsets):
			summation_layer[j] = np.sum(
				rbf(test_point, subset[0], sigma)) / subset[0].shape[0] 
		
		predictions[i] = np.argmax(summation_layer)
		i = i + 1
	
	return predictions


def print_metrics(y_test, predictions):

	print('Confusion Matrix')
	print(confusion_matrix(y_test, predictions))
	print('Accuracy: {}'.format(accuracy_score(y_test, predictions)))
	#print('Precision: {}'.format(precision_score(y_test, predictions, average = 'micro')))
	#print('Recall: {}'.format(recall_score(y_test, predictions, average = 'micro')))
	

if __name__ == '__main__':
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

	data = read_data.create_data(train_csv_filename, test_csv_filename)

	predictions = PNN(data)

	print_metrics(data['y_test'], predictions)

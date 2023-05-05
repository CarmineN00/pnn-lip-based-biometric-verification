import numpy as np
import read_data
import pandas
import os

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

	num_testset = data['x_test'].shape[0]
	labels = np.unique(data['y_train'])
	num_class = len(labels)

	sigma = 10

	x_train_subsets = subset_by_class(data, labels)

	summation_layer = np.zeros(num_class)
	predictions = np.zeros(num_testset)

	for i, test_point in enumerate(data['x_test']):
		
		for j, subset in enumerate(x_train_subsets):
			summation_layer[j] = np.sum(
				rbf(test_point, subset[0], sigma)) / subset[0].shape[0] 
		
		predictions[i] = np.argmax(summation_layer)
	
	return predictions


def print_metrics(y_test, predictions):

	print('Confusion Matrix')
	print(confusion_matrix(y_test, predictions))
	print('Accuracy: {}'.format(accuracy_score(y_test, predictions)))
	print('Precision: {}'.format(precision_score(y_test, predictions, average = 'micro')))
	print('Recall: {}'.format(recall_score(y_test, predictions, average = 'micro')))
	

if __name__ == '__main__':
	csv_filename = "dataset.csv"
	
	if not os.path.exists(csv_filename):
		read_data.create_csv(csv_filename)

	#data = pandas.read_csv(csv_filename)

	predictions = PNN(data)
	print_metrics(data['y_test'], predictions)

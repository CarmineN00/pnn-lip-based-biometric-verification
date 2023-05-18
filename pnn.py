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
    # Ridimensiona l'array 'centre' in una matrice bidimensionale con forma (1, n),
    # dove n è il numero di elementi nel vettore 'centre'.
    centre = centre.reshape(1, -1)

    # Calcola la distanza euclidea al quadrato tra 'centre' e 'x'
    # mediante sottrazione elemento per elemento, elevazione al quadrato e somma lungo l'asse 1.
    distanza_quad = -np.sum((centre - x) ** 2, axis=1)

    # Divide la distanza quadrata per (2 * sigma * sigma).
    distanza_quad_normalizzata = distanza_quad / (2 * sigma * sigma)

    # Calcola l'esponenziale della distanza quadrata normalizzata.
    esponenziale = np.exp(distanza_quad_normalizzata)

    # Somma tutti gli elementi dell'esponenziale.
    rbf = np.sum(esponenziale)

    return rbf

def laplacian(centre, x, sigma):
    # Ridimensiona l'array 'centre' in una matrice bidimensionale con forma (1, n),
    # dove n è il numero di elementi nel vettore 'centre'.
    centre = centre.reshape(1, -1)

	# Calcola la distanza euclidea al quadrato tra 'centre' e 'x'
    # mediante sottrazione elemento per elemento, elevazione al quadrato e somma lungo l'asse 1.
    distance_squared = np.sum((centre - x) ** 2, axis=1)
    
    # Calcola la distanza euclidea prendendo la radice quadrata
    # della distanza euclidea al quadrato.
    distance = np.sqrt(distance_squared)
    
	# Calcola la funzione Laplaciana mediante l'esponenziale del negativo
    # della distanza divisa per 'sigma'.
    laplacian = np.exp(-distance / sigma)
    return laplacian

def uniform(centre, x, sigma):
    # Ridimensiona l'array 'centre' in una matrice bidimensionale con forma (1, n),
    # dove n è il numero di elementi nel vettore 'centre'.
    centre = centre.reshape(1, -1)
    
	# Calcola la norma euclidea della differenza tra 'centre' e 'x'.
    distance = np.linalg.norm(centre - x, axis=1)
    
	# Crea un nuovo array 'uniform' inizializzato con 1 se la distanza è minore o uguale a 'sigma',
    # altrimenti viene impostato a 0. Questo genera una funzione uniforme che assume valore 1
    # all'interno di un certo raggio e valore 0 al di fuori di esso.# Crea un nuovo array 'uniform' inizializzato con 1 se la distanza è minore o uguale a 'sigma',
    # altrimenti viene impostato a 0. Questo genera una funzione uniforme che assume valore 1
    # all'interno di un certo raggio e valore 0 al di fuori di esso.
    uniform = np.where(distance <= sigma, 1, 0)
    
    return uniform

def epanechnikov(centre, x, sigma):
    # Ridimensiona l'array 'centre' in una matrice bidimensionale con forma (1, n),
    # dove n è il numero di elementi nel vettore 'centre'.
    centre = centre.reshape(1, -1)
    
    # Calcola la distanza euclidea al quadrato tra 'centre' e 'x'
    # mediante sottrazione elemento per elemento, elevazione al quadrato e somma lungo l'asse 1.
    distanza_quad = np.sum((centre - x) ** 2, axis=1)
    
    # Calcola la distanza euclidea prendendo la radice quadrata
    # della distanza euclidea al quadrato.
    distanza = np.sqrt(distanza_quad)
    
    # Crea un nuovo array 'epanechnikov' che viene calcolato utilizzando la formula
    # del kernel di Epanechnikov. La formula prevede un fattore moltiplicativo costante di 0.75,
    # seguito da (1 - (distanza_quad / (sigma ** 2))). Questa formula assegna un valore ponderato
    # alla funzione in base alla distanza dalla posizione centrale, con un massimo raggiunto quando
    # la distanza è zero (1 - 0 = 1) e una diminuzione quadratica man mano che la distanza aumenta.
    epanechnikov = np.where(distanza <= sigma, 0.75 * (1 - distanza_quad / (sigma ** 2)), 0)
    
    return epanechnikov

def triangle(centre, x, sigma):
    # Ridimensiona l'array 'centre' in una matrice bidimensionale con forma (1, n),
    # dove n è il numero di elementi nel vettore 'centre'.
    centre = centre.reshape(1, -1)
    
    # Calcola la norma euclidea della differenza tra 'centre' e 'x'.
    distance = np.linalg.norm(centre - x, axis=1)
    
    # Crea un nuovo array 'triangle' che viene calcolato utilizzando la formula
    # della funzione triangolare. La formula prevede un valore costante di 1 meno
    # la distanza normalizzata rispetto a 'sigma'. In altre parole, misura la distanza
    # relativa tra il punto e il centro in proporzione a 'sigma', con un valore massimo
    # di 1 raggiunto quando la distanza è zero e una diminuzione lineare man mano che
    # la distanza aumenta.
    triangle = np.where(distance <= sigma, 1 - distance / sigma, 0)
    
    return triangle


def subset_by_class(data, labels):
	x_train_subsets = []

	for l in tqdm(labels, desc="Subset_by_class", ncols=100):

		#Per ogni riga in data y train, verificare se tale riga è uguale alla l corrente
		#In esito positivo, inserire l'indice della riga in indices

		indices = []
		for j, row in enumerate(data['y_train']):
			if (np.array_equal(row,l)):
				indices.append(j)
		
		# print("\n\nPer la label", l, " gli indici sono:", indices)
		x_train_subsets.append(data['x_train'][indices, :])

	return x_train_subsets


def PNN(data, kernel_name):
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

	#print("Num classes: ",num_class)

	sigma = 10

	x_train_subsets = subset_by_class(data, labels)

	#print("Lenght of x_train_subsets: ",len(x_train_subsets))
	#for a,subset in enumerate(x_train_subsets):
		#print("Subset ", a, "has shape: ",np.shape(subset))

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
				if (kernel_name == "rbf"):
					summation_layer[j] = np.sum(rbf(test_point, subset[0], sigma)) / subset[0].shape[0]
				elif (kernel_name == "laplacian"):
					summation_layer[j] = np.sum(laplacian(test_point, subset[0], sigma)) / subset[0].shape[0]
				elif (kernel_name == "uniform"):
					summation_layer[j] = np.sum(uniform(test_point, subset[0], sigma)) / subset[0].shape[0]
				elif (kernel_name == "epanechnikov"):
					summation_layer[j] = np.sum(epanechnikov(test_point, subset[0], sigma)) / subset[0].shape[0]
				elif (kernel_name == "triangle"):
					summation_layer[j] = np.sum(triangle(test_point, subset[0], sigma)) / subset[0].shape[0]
			else:
				#print("\tSubset shape", j, "shape: ", np.shape(subset))
				summation_layer[j] = 0
			#print("Summation layer ", j, ": ", summation_layer[j])
	
		predictions[i] = np.argmax(summation_layer)
		
		i = i + 1
	
	return predictions



def print_metrics(y_test, predictions):
	#print("Y test:", np.shape(y_test))
	#print("Predictions: ",np.shape(predictions))

	#print('Confusion Matrix')
	#print(confusion_matrix(y_test, predictions))
	#print('Accuracy: {}'.format(accuracy_score(y_test, predictions)))
	#print('Precision: {}'.format(precision_score(y_test, predictions, average='macro', zero_division=1)))
	#print('Recall: {}'.format(recall_score(y_test, predictions, average='macro', zero_division=1)))
	#print('Precision: {}'.format(precision_score(y_test, predictions, average = 'micro')))
	#print('Recall: {}'.format(recall_score(y_test, predictions, average = 'micro')))

	scores = {
		"accuracy": accuracy_score(y_test, predictions),
		"precision": precision_score(y_test, predictions, average='macro', zero_division=1),
		"recall": recall_score(y_test, predictions, average='macro', zero_division=1)
	}

	return scores
	

if __name__ == '__main__':

	##########################################################################
	# OGNI METODO DI READ_DATA DEVE ORA ESSERE CHIAMATO DA read_data #
	##########################################################################

	train_csv_filename = "train_dataset.csv"
	test_csv_filename = "test_dataset.csv"

	train_directory = "Dataset/Train"
	test_directory = "Dataset/Test"

	kernel = "rbf"
	num_frames = "20"

	# Il dataset viene generato esclusivamente se non presente nella current working directory
	if not os.path.exists(train_csv_filename) or not os.path.exists(test_csv_filename):
		# Creazione del dataset di training
		read_data.create_csv(train_csv_filename, train_directory, kernel, num_frames)
		# Creazione del dataset di test
		read_data.create_csv(test_csv_filename, test_directory, kernel, num_frames)

	datasets = [train_csv_filename, test_csv_filename]

	data = read_data.create_data_correctly(train_csv_filename, test_csv_filename)

	predictions = PNN(data, kernel)

	#print(type(predictions))

	pd.DataFrame(predictions).to_csv("predictions.csv")

	print_metrics(data['y_test_before_ohe'], predictions)

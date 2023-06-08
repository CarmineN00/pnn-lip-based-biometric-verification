# Vanno prelevati 1200 frame di un individuo e 1800 frame tra gli altri individui

# Per ogni video dell'individuo, dobbiamo prelevare 255 frame

# Per ogni video del non individuo, dobbiamo prelevare 7 frame

# In questo modo il dataset sarà bilanciato 40-60

import numpy as np
import read_data as rd
import os
import csv
import glob
from tqdm import tqdm
import cv2
import LipLandmarks as lp
import mediapipe as mp
import math
import random
import pnn as p

def create_csv(csv_filename,directory,type,main_label,frames_for_main,frames_for_others):
    main_counter = 0
    other_counter = 0
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        header_string = []

        for i in range(20):
            header_string.append("Feature " + str(i))

        header_string.append("Label")
        writer.writerow(header_string)

        if os.path.isdir(directory):
            files = glob.glob(directory + "/*.avi")

            for video in tqdm(files, desc=directory, ncols=100):
                video_label = str(''.join(video.split("\\")[1].split(".")[0].split("_")[:4]))

                frames_to_fetch = 0
                if video_label == main_label:
                      frames_to_fetch = frames_for_main
                else:
                      frames_to_fetch = frames_for_others

                res = ottieni_features_da_video(video,type,frames_to_fetch)

                #print(csv_filename, "-",video_label,"-",np.shape(res))

                if np.shape(res)[0]!=0:
                      num_rows = np.shape(res)[0]/20
                      res_split = np.array_split(res,num_rows)

                      for elem in res_split:
                            if video_label == main_label:
                                writer.writerow(np.append(elem, "1"))
                                main_counter = main_counter + 1
                            else:
                                writer.writerow(np.append(elem, "0"))
                                other_counter = other_counter + 1
    '''print("Sono state prelevate",main_counter,"righe per il soggetto",main_label)
    print("Sono state prelevate",other_counter,"righe per gli altri soggetti")'''

def ottieni_features_da_video(filename,type,num_frames):

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    cap = cv2.VideoCapture(filename)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - num_frames - 1)

    ret, frame = cap.read()

    list_of_euclidean_distances = []

    while ret:

        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)

        iterator = iter(lp.lip_landmarks)
        for j in range(0, len(lp.lip_landmarks)):
            i = next(iterator)

            if results and results.multi_face_landmarks:
                # Primo elemento della tupla
                point1 = results.multi_face_landmarks[0].landmark[i[0]]
                node1_x = int(point1.x * width)
                node1_y = int(point1.y * height)
                if (type == "DistanzeEuclidee2D"):
                    cv2.circle(frame, (node1_x, node1_y), 2, (255, 255, 255), -1)

                # Secondo elemento della tupla
                point2 = results.multi_face_landmarks[0].landmark[i[1]]
                node2_x = int(point2.x * width)
                node2_y = int(point2.y * height)
                if (type == "DistanzeEuclidee2D"):
                    cv2.circle(frame, (node2_x, node2_y), 2, (255, 255, 255), -1)

                # Calcolo della distanza euclidea tra i punti
                if (type == "DistanzeEuclidee2D"):
                    d = math.sqrt((node2_x - node1_x) ** 2 + (node2_y - node1_y) ** 2)
                elif (type == "DistanzeEuclideeNormalizzate2D"):
                    d = math.sqrt((point2.x - point1.y) ** 2 + (point2.y - point1.y) ** 2)
                elif (type == "CityBlock3D"):
                    d = abs(point1.x - point2.x) + abs(point1.y - point2.y) + abs(point1.z - point2.z)

                list_of_euclidean_distances.append(d)

    return list_of_euclidean_distances

def PNN(data, kernel_name):
	num_test_set = data['x_test'].shape[0]

	#Anche questo va modificato, perchè non è detto che in y_train ci sia almeno un sample di tutte e 256 persone
	# labels = np.unique(data['y_train'], axis=0)

	labels = []

	# Inizializza una matrice vuota per contenere i vettori one-hot-encoded
	one_hot_matrix = np.zeros((2, 2))

	# Genera i vettori one-hot-encoded
	for i in range(2):
		one_hot_matrix[i][i] = 1

	for row in one_hot_matrix:
		labels.append(list(row))
	
	#print(labels)
 
	num_class = len(labels)

	#print("Num classes: ",num_class)

	sigma = 10

	x_train_subsets = p.subset_by_class(data, labels)

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
					summation_layer[j] = np.sum(p.rbf(test_point, subset[0], sigma)) / subset[0].shape[0]
				elif (kernel_name == "laplacian"):
					summation_layer[j] = np.sum(p.laplacian(test_point, subset[0], sigma)) / subset[0].shape[0]
				elif (kernel_name == "uniform"):
					summation_layer[j] = np.sum(p.uniform(test_point, subset[0], sigma)) / subset[0].shape[0]
				elif (kernel_name == "epanechnikov"):
					summation_layer[j] = np.sum(p.epanechnikov(test_point, subset[0], sigma)) / subset[0].shape[0]
				elif (kernel_name == "triangle"):
					summation_layer[j] = np.sum(p.triangle(test_point, subset[0], sigma)) / subset[0].shape[0]
			else:
				#print("\tSubset shape", j, "shape: ", np.shape(subset))
				summation_layer[j] = 0
			#print("Summation layer ", j, ": ", summation_layer[j])
	
		predictions[i] = np.argmax(summation_layer)
		
		i = i + 1
	
	return predictions

video_labels = rd.old_get_labels(["Dataset/Train","Dataset/Test"])
unique_labels = np.unique(video_labels)
num_people = len(unique_labels)

#print("Ci sono",num_people,"persone differenti nel Dataset!")

#Variabili indipendenti
proportion = (0.2,0.8)
num_frames_of_main_subject = 1200

#Variabili dipendenti
num_frames_of_other_subjects = int(np.floor((num_frames_of_main_subject*proportion[1])/proportion[0]))

num_frames_of_main_subject_in_test = int(np.floor(num_frames_of_main_subject*0.2))
num_frames_of_main_subject_in_train = num_frames_of_main_subject - num_frames_of_main_subject_in_test
num_frames_of_other_subjects_in_test = int(np.floor(num_frames_of_other_subjects*0.2))
num_frames_of_other_subjects_in_train = num_frames_of_other_subjects - num_frames_of_other_subjects_in_test

'''print("Dobbiamo prelevare ",num_frames_of_main_subject,"frames dal main subject di cui\n\t",num_frames_of_main_subject_in_test,"dalla cartella di Test\n\t",num_frames_of_main_subject_in_train,"dalla cartella di Train")
print("Dobbiamo prelevare ",num_frames_of_other_subjects,"frames dagli altri subjects di cui\n\t",num_frames_of_other_subjects_in_test,"dalla cartella di Test\n\t",num_frames_of_other_subjects_in_train,"dalla cartella di Train")
'''
# Creiamo i CSV

label_main_subject = random.choice(unique_labels)
     
print("Il fortunato è :",label_main_subject)

frames_for_test_video_of_main_subject = num_frames_of_main_subject_in_test
frames_for_test_video_of_other_subject = int(np.floor(num_frames_of_other_subjects_in_test/(num_people-1)))

frames_for_train_video_of_main_subject = num_frames_of_main_subject_in_train
frames_for_train_video_of_other_subject = int(np.floor(num_frames_of_other_subjects_in_train/(num_people-1)))

'''print("Da ogni video di TEST del MAIN subject vanno prelevati",frames_for_test_video_of_main_subject, "frames")
print("Da ogni video di TEST degli OTHER subject vanno prelevati",frames_for_test_video_of_other_subject)
print("Da ogni video di TRAIN del MAIN subject vanno prelevati",frames_for_train_video_of_main_subject)
print("Da ogni video di TRAIN degli OTHER subject vanno prelevati",frames_for_train_video_of_other_subject)'''

metriche = ["DistanzeEuclidee2D", "DistanzeEuclideeNormalizzate2D", "CityBlock3D"]

csv_filename_test = "VERIFY_"+metriche[0]+"_"+label_main_subject+"_test.csv"
create_csv(csv_filename_test,"Dataset/Test",metriche[0],label_main_subject,frames_for_test_video_of_main_subject,frames_for_test_video_of_other_subject)
#print(csv_filename_test," creato con successo!")

csv_filename_train = "VERIFY_"+metriche[0]+"_"+label_main_subject+"_train.csv"
create_csv(csv_filename_train,"Dataset/Train",metriche[0],label_main_subject,frames_for_train_video_of_main_subject,frames_for_train_video_of_other_subject)
#print(csv_filename_train," creato con successo!")

data = rd.create_data_correctly(csv_filename_train,csv_filename_test)

kernel_names = ["rbf", "laplacian", "uniform", "epanechnikov", "triangle"]

predictions = PNN(data, kernel_names[0])

scores = p.print_metrics(data['y_test_before_ohe'], predictions)

print("\n\nSUBJECT",label_main_subject,"\nAccuracy: "+str(scores["accuracy"])+"\nPrecision: "+str(scores["precision"])+"\nRecall: "+str(scores["recall"])+"\n\n")
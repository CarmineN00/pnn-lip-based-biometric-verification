import pandas
import numpy as np
import os, glob, cv2
import math
import mediapipe as mp
import pandas
import LipLandmarks as lp
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from tqdm import tqdm
from scipy.spatial.distance import cityblock
import read_data
import pnn as p

num_frames = 20

if __name__ == "__main__":
    metriche = ["DistanzeEuclidee2D", "DistanzeEuclideeNormalizzate2D", "CityBlock3D"]
    experiments = ["fullmesh", "twenty", "fullmesh_spaced", "twenty_spaced"]

    for metrica in metriche:
        for experiment in experiments:
            csv_filename = metrica + "_" + str(num_frames) + "_" + experiment + "_"
        
            test_csv_file = csv_filename + "test_dataset.csv"
            if not os.path.exists(test_csv_file):
                read_data.create_csv(test_csv_file, "Dataset/Test", metrica, num_frames, experiment)
            else:
                print(test_csv_file, "già esiste")
        
            train_csv_file = csv_filename + "train_dataset.csv"
            if not os.path.exists(train_csv_file):
                read_data.create_csv(train_csv_file, "Dataset/Train", metrica, num_frames, experiment)
            else:
                print(train_csv_file, "già esiste")

    kernel_names = ["rbf", "laplacian", "uniform", "epanechnikov", "triangle"]
    
    file_rows = ""

    for kernel in kernel_names:
        for metrica in metriche:
            for experiment in experiments:
                csv_filename = metrica + "_" + str(num_frames) + "_" + experiment + "_"
                test_csv_file = csv_filename + "test.csv"
                train_csv_file = csv_filename + "train.csv"
                
                data = read_data.create_data_correctly(train_csv_file, test_csv_file)
                predictions = p.PNN(data,kernel)

                to_print = "Kernel: "+kernel+"\nMetric: "+metrica+"\nExperiment: "+experiment
                scores = p.print_metrics(data['y_test_before_ohe'], predictions)
                to_print+="\nAccuracy: "+str(scores["accuracy"])+"\nPrecision: "+str(scores["precision"])+"\nRecall: "+str(scores["recall"])+"\n\n"
                print(to_print)
                file_rows += to_print

    with open('results.txt', 'w') as file:
        file.write(file_rows)

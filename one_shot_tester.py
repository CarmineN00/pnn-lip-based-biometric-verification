import read_data as rd
import pnn as p
import os
import shutil

num_frames = 50
experiment = "twentytwo_spaced_behavioural"
type = "DistanzeEuclidee2D"

train_csv_filename = type+"_"+str(num_frames)+"_"+experiment+"_train_dataset.csv"
test_csv_filename = type+"_"+str(num_frames)+"_"+experiment+"_test_dataset.csv"
	
configuration = train_csv_filename.split("_train_dataset.csv")[0]

print("[",configuration,"] --- Testing ...\n")

train_directory = "Dataset/Train"
test_directory = "Dataset/Test"

if not os.path.exists("datasets/"+train_csv_filename):
    print("Creando il file di addestramento.")
    rd.create_csv(train_csv_filename,train_directory,"DistanzeEuclidee2D",num_frames,experiment)

else:
    print("Il file di addestramento esiste.")

if not os.path.exists("datasets/"+test_csv_filename):
    print("Creando il file di test.")
    rd.create_csv(test_csv_filename,test_directory,"DistanzeEuclidee2D",num_frames,experiment)
else:
    print("Il file di test esiste.")


shutil.move(train_csv_filename, "datasets")
shutil.move(test_csv_filename, "datasets")

kernel = "rbf"

data = rd.create_data_correctly("datasets/"+train_csv_filename,"datasets/"+test_csv_filename)

predictions = p.PNN(data, kernel)

scores = p.print_metrics(data['y_test_before_ohe'], predictions)
to_print="\n"+configuration+"\nAccuracy: "+str(scores["accuracy"])+"\nPrecision: "+str(scores["precision"])+"\nRecall: "+str(scores["recall"])+"\n\n"
print(to_print)
import read_data as rd
import pnn as p
import os
import shutil
from plyer import notification

def send_notification(title, message):
    notification.notify(
        title=title,
        message=message,
        timeout=5  # Durata della notifica in secondi
    )

num_frames = 300
experiment = "twentytwo_spaced_behavioural"
type = "DistanzeEuclidee2D"
kernel = "laplacian"

train_csv_filename = type+"_"+str(num_frames)+"_"+experiment+"_train_dataset.csv"
test_csv_filename = type+"_"+str(num_frames)+"_"+experiment+"_test_dataset.csv"
	
configuration = train_csv_filename.split("_train_dataset.csv")[0]

print("[",configuration,"] --- Testing ...\n")

train_directory = "Dataset/Train"
test_directory = "Dataset/Test"

if not os.path.exists("datasets/"+train_csv_filename):
    print("Creando il file di addestramento.")
    rd.create_csv(train_csv_filename,train_directory,"DistanzeEuclidee2D",num_frames,experiment)
    shutil.move(train_csv_filename, "datasets")
else:
    print("Il file di addestramento esiste.")

if not os.path.exists("datasets/"+test_csv_filename):
    print("Creando il file di test.")
    rd.create_csv(test_csv_filename,test_directory,"DistanzeEuclidee2D",num_frames,experiment)
    shutil.move(test_csv_filename, "datasets")
else:
    print("Il file di test esiste.")

data = rd.create_data_correctly("datasets/"+train_csv_filename,"datasets/"+test_csv_filename)

predictions = p.PNN(data, kernel)

scores = p.print_metrics(data['y_test_before_ohe'], predictions)
to_print="\n"+configuration+"\nAccuracy: "+str(scores["accuracy"])+"\nPrecision: "+str(scores["precision"])+"\nRecall: "+str(scores["recall"])+"\n\n"
print(to_print)

send_notification("Hey", "La PNN ha terminato, vieni a vedere")
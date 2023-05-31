import read_data as rd
import pnn as p


train_csv_filename = "DistanzeEuclidee2D_300_twentytwo_behavioural_train_dataset.csv"
test_csv_filename = "DistanzeEuclidee2D_300_twentytwo_behavioural_test_dataset.csv"
	
configuration = train_csv_filename.split("_train_dataset.csv")[0]

print("[",configuration,"] --- Testing ...\n")

train_directory = "Dataset/Train"
test_directory = "Dataset/Test"

kernel = "rbf"
num_frames = "20"

data = rd.create_data_correctly(train_csv_filename, test_csv_filename)

predictions = p.PNN(data, kernel)

scores = p.print_metrics(data['y_test_before_ohe'], predictions)
to_print="\n"+configuration+"\nAccuracy: "+str(scores["accuracy"])+"\nPrecision: "+str(scores["precision"])+"\nRecall: "+str(scores["recall"])+"\n\n"
print(to_print)
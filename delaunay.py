import read_data as rd
import os
import pnn as p

if __name__ == "__main__":


    test_csv_file = "delaunay_test.csv"
    if not os.path.exists(test_csv_file):
        rd.create_csv(test_csv_file, "Dataset/Test", "", 20, "delaunay")
    else:
        print(test_csv_file, "già esiste")

    train_csv_file = "delaunay_train.csv"
    if not os.path.exists(train_csv_file):
        rd.create_csv(train_csv_file, "Dataset/Train", "", 20, "delaunay")
    else:
        print(train_csv_file, "già esiste")

    data = rd.create_data_correctly(train_csv_file, test_csv_file)
    predictions = p.PNN(data,"rbf")

    scores = p.print_metrics(data['y_test_before_ohe'], predictions)
    print("\nAccuracy: "+str(scores["accuracy"])+"\nPrecision: "+str(scores["precision"])+"\nRecall: "+str(scores["recall"])+"\n\n")
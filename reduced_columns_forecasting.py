import read_data as rd
import pnn as p
import os

if __name__ == "__main__":
    
    columns = 15

    test_csv_file = "spaced_test_dataset.csv"
    if not os.path.exists(test_csv_file):
        rd.create_csv(test_csv_file, "Dataset/Test", "DistanzeEuclidee2D", 20, "spaced")
    else:
        print(test_csv_file, "già esiste")
    
    train_csv_file = "spaced_train_dataset.csv"
    if not os.path.exists(train_csv_file):
        rd.create_csv(train_csv_file, "Dataset/Train", "DistanzeEuclidee2D", 20, "spaced")
    else:
        print(train_csv_file, "già esiste")

    for columns in range(5, 21):

        print("Testing only using",columns,"columns")
    
        data = rd.create_data_correctly("spaced_train_dataset.csv","spaced_test_dataset.csv")

        '''print("Printing stats for data!")
        rd.print_stats(data)
        print("\n")'''

        reduced_data = rd.reduce_data(data, columns)

        '''print("Printing stats for reduced_data")
        rd.print_stats(reduced_data)
        print("\n")'''

        predictions = p.PNN(reduced_data,"rbf")

        scores = p.print_metrics(reduced_data['y_test_before_ohe'], predictions)
        to_print="\nColumns: "+str(columns)+"\nAccuracy: "+str(scores["accuracy"])+"\nPrecision: "+str(scores["precision"])+"\nRecall: "+str(scores["recall"])+"\n\n"
        print(to_print)
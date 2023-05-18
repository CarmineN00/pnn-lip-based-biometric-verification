import read_data
import numpy as np
import pandas

data, raw_data = read_data.create_data("train_dataset.csv","test_dataset.csv")

print("Shape of x_train:", np.shape(data['x_train']))
print("Shape of y_train (OHE):", np.shape(data['y_train']))
print("Shape of x_test:", np.shape(data['x_test']))
print("Shape of y_test (OHE):", np.shape(data['y_test']))

print("\n\nShape of y_test RAW:", np.shape(raw_data['y_test']))
print("Shape of y_train RAW:", np.shape(raw_data['y_train']))
y_full = np.concatenate((raw_data['y_test'],raw_data['y_train']), axis=0)
print("Shape of y_full :", np.shape(y_full))
print("Numero individui distinti totali trovati in y_full: ", len(np.unique(y_full)))

#Il one hot encoding va fatto su questo numero, che il numero TOTALE di individui distinti in tutte le cartelle
video_label_list = read_data.get_labels(["Dataset/Test", "Dataset/Train"])
num_distinct_people = len(np.unique(video_label_list))
print("\nNumero reali individui distinti: ",num_distinct_people)

def ohe_correct(y, width):
    y_ohe = np.zeros((y.shape[0], width))
    
    for i in range(y.shape[0]):
        y_ohe[i, y[i]] = 1

    pandas.DataFrame(y_ohe).to_csv("ohe_classes.csv")

    return y_ohe

y_test_ohe_correct = ohe_correct(raw_data['y_test'], num_distinct_people)
y_train_ohe_correct = ohe_correct(raw_data['y_train'], num_distinct_people)

print("\n\nShape of y_test_ohe_correct", np.shape(y_test_ohe_correct))
print("Shape of y_train_ohe_correct", np.shape(y_train_ohe_correct))

#Le righe, aventi le label crude "11141" etc... devono essere correttamente labellate in numeri che vanno da 0 a 255
#Soltanto dopo è possibile fare il one-hot-encoding associando ad ogni numero j da 0 a 255 un vettore di dimensione 256 avente soli 0 con un 1 in posizione j

#Va creata inizialmente un hashmap che associa alle label stringhe univocamente un numero da 0 a 255

def create_hashmap_from_ugly_labels_to_numbers(list):
    hashmap = {}
    i = 0
    for elem in list:
        if hashmap.get(elem) is None:
            hashmap[elem] = i
            i = i+1
    return hashmap

hashmap = create_hashmap_from_ugly_labels_to_numbers(video_label_list)
print("\nHashMap\n", hashmap)

# Ora che ho questa hashmap posso finalmente fare in modo che ogni riga, che sia in test o in train
# abbia il suo vero id univoco da 0 a 255 mantenendo la consistenza tra gli individui, cioè uno stesso individuo
# che sia relativo ad una riga relativa al test o al train deve avere lo stesso id da 0 a 255!

# Ora prelevo il dataset test e il dataset train raw, con anche le label bruttine, e converto le label bruttine nell'id da 0 a 255

#Prima per il dataset train

df_train = pandas.read_csv("train_dataset.csv")
x_train = df_train.iloc[:, :-1].values
y_train = df_train['Label'].values

print("Before converting y_train: ", y_train)

for j in range (len(y_train)):
    print(y_train[j], "->", hashmap[str(y_train[j])])
    y_train[j] = hashmap[str(y_train[j])]

print("After converting y_train: ", y_train)

#Ora, la stessa conversione anche per il dataset test

df_test = pandas.read_csv("test_dataset.csv")
df_test.sort_values(by=['Label'])

x_test = df_test.iloc[:, :-1].values
y_test = df_test['Label'].values

print("Before converting y_test: ", y_test)

for j in range (len(y_test)):
    print(y_test[j], "->", hashmap[str(y_test[j])])
    y_test[j] = hashmap[str(y_test[j])]

print("After converting y_test: ", y_test)

# Ora e soltanto ora possiamo usare il one-hot-encoding!
# Rinominiamo le variabili per una maggior chiarezza

x_test_clean = x_test
y_test_clean = y_test
x_train_clean = x_train
y_train_clean = y_train

y_test_clean_ohe = ohe_correct(y_test_clean, num_distinct_people)
y_train_clean_ohe = ohe_correct(y_train_clean, num_distinct_people)

print("Shape of x_test_clean:", np.shape(x_test_clean))
print("Shape of x_train_clean:", np.shape(x_train_clean))
print("Shape of y_test_clean_ohe:", np.shape(y_test_clean_ohe))
print("Shape of y_train_clean_ohe:", np.shape(y_train_clean_ohe))

# x_test_clean, x_train_clean, y_test_clean_ohe e y_train_clean_ohe sono i dati che devono essere passati alla PNN, CORRETTI!

def get_labels(csvs):
    video_labels = []
    for csv in csvs:
        df_train = pandas.read_csv(csv)
        y_train = df_train['Label'].values
        for elem in y_train:
            video_labels.append(elem)
    return video_labels

test_video_labels = get_labels(["test_dataset.csv","train_dataset.csv"])
print(test_video_labels)
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import read_data
from sklearn import svm

estimator = "rf"

if __name__ == '__main__':
    train_csv_filename = "train_dataset.csv"
    test_csv_filename = "test_dataset.csv"

    train_directory = "Dataset/Train"
    test_directory = "Dataset/Test"

    data = read_data.create_data_correctly(train_csv_filename, test_csv_filename)

    X = data['x_train']
    y = data['y_train_before_ohe']

    X_test = data['x_test']
    y_test = data['y_test_before_ohe']

    if estimator == "svc":
        print("SVM")
        current_params = {
            'C': [0.1, 1, 10, 100, 1000], 
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['linear', 'rbf', 'poly']
        }
        svm = svm.SVC()
        current_estimator = svm
    elif estimator == "rf":
        print("Random Forest")
        current_params = {
            'n_estimators': [48,80],
            'max_features': ['auto','sqrt'],
            'max_depth': [2,4],
            'min_samples_split': [2,5],
            'min_samples_leaf': [1,2],
            'bootstrap': [True, False]
        }
        random_forest = RandomForestClassifier()
        current_estimator = random_forest


    grid = GridSearchCV(
        estimator = current_estimator, 
        param_grid = current_params, 
        cv = 3, 
        n_jobs = -1,
        verbose=10
    )

    grid.fit(X, y)
	
    print(grid.best_params_)

    accuracy_orc = float("{:.2f}".format(grid.score(X_test, y_test) * 100))

    print(accuracy_orc)
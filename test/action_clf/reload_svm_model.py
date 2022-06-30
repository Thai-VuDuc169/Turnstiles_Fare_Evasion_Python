import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

GLOBAL_RANDOM_SEED = 42
CSV_DATASET_PATH = r"/home/thaivu169/Projects/Turnstiles_Fare_Evasion_Python/scripts/SVM/RT_v3.csv"
FILE_NAME = r"/home/thaivu169/Projects/Turnstiles_Fare_Evasion_Python/scripts/SVM/svm_action_recog.sav"

#load dataset 
dataset_df = pd.read_csv(CSV_DATASET_PATH) 
# label = 1 (walking), label = 0 (not walking)
dataset_np = np.asarray(dataset_df, dtype= 'float64', order= 'C')
np.random.shuffle(dataset_np)
X = dataset_np[:, :-1]
Y = dataset_np[:, -1]
# split dataset to traning, testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state= GLOBAL_RANDOM_SEED)
print ("the number of traning set is: " + str(y_train.shape[0]))
print ("the number of testing set is: " + str(y_test.shape[0]))


loaded_model = pickle.load(open(FILE_NAME, 'rb'))
loaded_result = loaded_model.score(X_test, y_test)
print(loaded_result)
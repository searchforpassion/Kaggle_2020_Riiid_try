import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#directory_path = '../input/pretrained-for-riiid/'
directory_path = ''
with open(directory_path + 'user_ele_dict.pkl', 'rb') as f:
    user_ele_dict = pickle.load(f)

with open(directory_path + 'user_num_dict.pkl', 'rb') as f:
    user_num_dict = pickle.load(f)

with open(directory_path + 'ques_ele_vec.pkl', 'rb') as f:
    ques_ele_vec = pickle.load(f)

with open(directory_path + 'ques_num_vec.pkl', 'rb') as f:
    ques_num_vec = pickle.load(f)  

question_dtype = {
    'question_id':'int16',
    'tags':'object'
}
questions_data = pd.read_csv(directory_path + 'questions.csv',
                             usecols = question_dtype.keys(), 
                             dtype = question_dtype)

questions_data.tags.fillna('92',inplace=True)

def gen_vec(row):
    row['vec'] = np.zeros(188)
    index_list = row.tags.split()
    for index_ in index_list:
        row.vec[int(index_)] = 1.0
    return row

questions_data = questions_data.apply(gen_vec, axis='columns')

train_dtypes_dict = {
    "row_id": "int64",
    #"timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "content_type_id": "int8",
    #"task_container_id": "int16",
    #"user_answer": "int8",
    "answered_correctly": "int8",
    #"prior_question_elapsed_time": "float32", 
    #"prior_question_had_explanation": "boolean"
}

train_data = pd.read_csv(directory_path + "train.csv",
                         nrows=10**5,
                         usecols = train_dtypes_dict.keys(),
                         dtype=train_dtypes_dict,
                         #index_col = 0,
                        )
train_data_q = train_data[train_data.content_type_id == 0]

del train_data

def pre_data(row_data, ele_dict, num_dict, ques_ele=ques_ele_vec,ques_num=ques_num_vec,q_data=questions_data):
    m = len(row_data)
    X = np.ones((m,188*3)) * 0.25
    y = np.zeros(m)
    i = 0
    ave = ques_ele/ques_num
    for index, row in tqdm(row_data.iterrows()):
        mask = q_data.vec[row.content_id]
        if row.user_id in ele_dict:
            X[i,:188] = np.nan_to_num(ele_dict[row.user_id]/num_dict[row.user_id],nan=0.25)
        X[i,188:188*2] = mask
        X[i,188*2:] = ave
        y[i] = row.answered_correctly
        i = i+1
    return X, y

from tensorflow import keras
#from tensorflow.keras import layers

reconstructed_model = keras.models.load_model("my_h5_model.h5")

test_sample = train_data_q[:10000]
X_test, y_test = pre_data(test_sample, user_ele_dict, user_num_dict)

predict_prob = reconstructed_model.predict(X_test)

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, predict_prob))
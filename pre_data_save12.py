import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

with open('user_ele_dict.pkl', 'rb') as f:
    user_ele_dict = pickle.load(f)

with open('user_num_dict.pkl', 'rb') as f:
    user_num_dict = pickle.load(f)

with open('ques_ele_vec.pkl', 'rb') as f:
    ques_ele_vec = pickle.load(f)

with open('ques_num_vec.pkl', 'rb') as f:
    ques_num_vec = pickle.load(f)

question_dtype = {
    'question_id':'int16',
    'tags':'object'
}
questions_data = pd.read_csv('questions.csv',
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

print("Start loading data.....")
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

data = pd.read_csv("train.csv",
                         #nrows=10**5,
                         usecols = train_dtypes_dict.keys(),
                         dtype=train_dtypes_dict,
                         #index_col = 0,
                        )

data = data[data.content_type_id == 0]
print("Loading completed")

users_i_counts = data.user_id.value_counts()
user_list = users_i_counts.keys()
print(user_list[:5])
print(users_i_counts.head())

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered")

ave = ques_ele_vec/ques_num_vec
def pre_data(user_list_, ele_dict=user_ele_dict, num_dict=user_num_dict, ave=ave,q_data=questions_data,data=data):
    m = len(user_list_)
    X = np.zeros((m,188*3))
    y = np.zeros(m)
    i = 0
    for user in tqdm(user_list_):
        user_data = data[data.user_id==user]
        mask = q_data.vec[user_data.content_id.values[-1]]
        X[i,:188] = np.nan_to_num(ele_dict[user]/num_dict[user],nan=0.25)
        X[i,188:188*2] = mask
        X[i,188*2:] = ave
        y[i] = user_data.answered_correctly.values[-1]
        i = i+1
    return X, y

X1,y1 = pre_data(user_list[300000:393382])
print(X1.shape)
print(y1.shape)
#print(X[0])
#print(y)
with open('X1.pkl', 'wb') as f:
    pickle.dump(X1, f)

with open('y1.pkl', 'wb') as f:
    pickle.dump(y1, f)

X2,y2 = pre_data(user_list[393382:])
print(X2.shape)
print(y2.shape)
#print(X[0])
#print(y)
with open('X2.pkl', 'wb') as f:
    pickle.dump(X2, f)

with open('y2.pkl', 'wb') as f:
    pickle.dump(y2, f)
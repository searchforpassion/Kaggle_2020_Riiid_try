import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

with open('user_ele_dict.pkl', 'rb') as f:
    user_ele_dict = pickle.load(f)

with open('user_num_dict.pkl', 'rb') as f:
    user_num_dict = pickle.load(f)

import copy
minus_ele_dict = copy.deepcopy(user_ele_dict)
minus_num_dict = copy.deepcopy(user_num_dict)

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

def minus_one(user_list_, ele_dict=minus_ele_dict, num_dict=minus_num_dict, q_data=questions_data,data=data):
    for user in tqdm(user_list_):
        user_data = data[data.user_id==user][-1:]
        num_dict[user] = num_dict[user] - q_data.vec[user_data.content_id.values[0]]
        ele_dict[user] = ele_dict[user] - q_data.vec[user_data.content_id.values[0]]*user_data.answered_correctly.values[0]

minus_one(user_list)

with open('minus_ele_dict.pkl', 'wb') as f:
    pickle.dump(minus_ele_dict, f)

with open('minus_num_dict.pkl', 'wb') as f:
    pickle.dump(minus_num_dict, f)
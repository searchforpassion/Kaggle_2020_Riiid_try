import numpy as np
import pandas as pd
from tqdm import tqdm

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

train_data = pd.read_csv("train.csv",
                         #nrows=10**6,
                         usecols = train_dtypes_dict.keys(),
                         dtype=train_dtypes_dict,
                         #index_col = 0,
                        )

train_data_q = train_data[train_data.content_type_id == 0]
print(len(train_data_q))
print(len(train_data_q.user_id.unique()))
print("Loading completed")

def cal_vec(train_row,ele_dict,num_dict,q_data=questions_data):
    num_dict[train_row.user_id] += q_data.vec[train_row.content_id]
    ele_dict[train_row.user_id] += q_data.vec[train_row.content_id] * train_row.answered_correctly

user_ele_dict = dict()
user_num_dict = dict()
ques_ele_vec = np.zeros(188)
ques_num_vec = np.zeros(188)
for index, row in tqdm(train_data_q.iterrows()):
    ques_ele_vec += questions_data.vec[row.content_id] * row.answered_correctly
    ques_num_vec += questions_data.vec[row.content_id]
    if row.user_id in user_ele_dict.keys():
        cal_vec(row,user_ele_dict,user_num_dict)
    else:
        user_ele_dict[row.user_id] = np.zeros(188)
        user_num_dict[row.user_id] = np.zeros(188)
        cal_vec(row,user_ele_dict,user_num_dict)

import pickle
with open('user_ele_dict.pkl', 'wb') as f:
    pickle.dump(user_ele_dict, f)

with open('user_num_dict.pkl', 'wb') as f:
    pickle.dump(user_num_dict, f)

with open('ques_ele_vec.pkl', 'wb') as f:
    pickle.dump(ques_ele_vec, f)

with open('ques_num_vec.pkl', 'wb') as f:
    pickle.dump(ques_num_vec, f)

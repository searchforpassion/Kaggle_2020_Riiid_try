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

train_data = pd.read_csv("train.csv",
                         nrows=10**6,
                         usecols = train_dtypes_dict.keys(),
                         dtype=train_dtypes_dict,
                         #index_col = 0,
                        )

train_data_q = train_data[train_data.content_type_id == 0]
print("Loading completed")

def predict_y(row_data, user_ele, user_num, ques_ele=ques_ele_vec, ques_num=ques_num_vec, q_data=questions_data):
    y = np.ones(len(row_data))*0.25
    i = 0
    for index, row in tqdm(row_data.iterrows()):
        mask = q_data.vec[row.content_id]
        ques_prob_vec = ques_ele/ques_num
        if row.user_id in user_ele.keys():
            borrow_index = (user_num[row.user_id] == 0)
            user_prob_vec = user_ele[row.user_id]/user_num[row.user_id]
            user_prob_vec[borrow_index] = ques_prob_vec[borrow_index]
            y[i] = sum(user_prob_vec * mask)/sum(mask)
        else:
            y[i] = sum(ques_prob_vec * mask)/sum(mask)
        i = i + 1
    return y

test_sample = train_data_q
prob = predict_y(test_sample,user_ele_dict, user_num_dict)
from sklearn.metrics import roc_auc_score
#print(prob)
#print(test_sample.answered_correctly.values)
print(roc_auc_score(test_sample.answered_correctly.values, prob))
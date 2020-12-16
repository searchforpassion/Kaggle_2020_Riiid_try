import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

with open('user_ele_dict.pkl', 'rb') as f:
    user_ele_dict = pickle.load(f)

with open('user_num_dict.pkl', 'rb') as f:
    user_num_dict = pickle.load(f)

# with open('../input/pretrained-for-riiid/ques_ele_vec.pkl', 'rb') as f:
#     ques_ele_vec = pickle.load(f)

# with open('../input/pretrained-for-riiid/ques_num_vec.pkl', 'rb') as f:
#     ques_num_vec = pickle.load(f)  

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

#train_data = data.sample(n=1000000, random_state = 1)

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered")

def pre_data(df, ele_dict=user_ele_dict, num_dict=user_num_dict, q_data=questions_data):
    m = len(df)
    X = np.zeros((m,188*2))
    y = np.zeros(m)
    i = 0
    for index, row in tqdm(df.iterrows()):
        mask = q_data.vec[row.content_id]
        X[i,:188] = np.nan_to_num(ele_dict[row.user_id]/num_dict[row.user_id],nan=0.25)
        X[i,188:] = mask
        y[i] = row.answered_correctly
        i = i+1
    return X, y

X_1, y_1 = pre_data(data[:10000000])

'''from sklearn.model_selection import train_test_split
X_big, X_test, y_big, y_test = train_test_split(X, y, test_size=0.1, random_state=40)
X_train, X_valid, y_train, y_valid = train_test_split(X_big, y_big, test_size=0.1, random_state=10)
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)

import lightgbm as lgb
params = {'objective': 'binary',
          'metric': 'auc',
          'seed': 2020,
          'learning_rate': 0.1, #default
          "boosting_type": "gbdt" #default
         }
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid)

model = lgb.train(
    params, lgb_train,
    valid_sets=[lgb_train, lgb_eval],
    verbose_eval=50,
    num_boost_round=10000,
    early_stopping_rounds=8
)

predict_prob = model.predict(X_test)
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, predict_prob))'''

with open('X_1.pkl', 'wb') as f:
    pickle.dump(X_1, f)

with open('y_1.pkl', 'wb') as f:
    pickle.dump(y_1, f)

# import joblib
# joblib.dump(my_model, 'lgb.pkl')

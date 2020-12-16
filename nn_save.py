import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
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

data = pd.read_csv(directory_path + "train.csv",
                         #nrows=10**6,
                         usecols = train_dtypes_dict.keys(),
                         dtype=train_dtypes_dict,
                         #index_col = 0,
                        )
data_q = data[data.content_type_id == 0]

del data
train_data_q = data_q
valid_data_q = data_q.sample(n=10**5)
del data_q
print(len(train_data_q),len(valid_data_q))

def pre_data(row_data, ele_dict, num_dict, ques_ele=ques_ele_vec,ques_num=ques_num_vec,q_data=questions_data):
    m = len(row_data)
    X = np.ones((m,188*3)) * 0.25
    y = np.zeros(m)
    i = 0
    ave = ques_ele/ques_num
    for index, row in row_data.iterrows():
        mask = q_data.vec[row.content_id]
        if row.user_id in ele_dict:
            X[i,:188] = np.nan_to_num(ele_dict[row.user_id]/num_dict[row.user_id],nan=0.25)
        X[i,188:188*2] = mask
        X[i,188*2:] = ave
        y[i] = row.answered_correctly
        i = i+1
    return X, y

X_valid, y_valid = pre_data(valid_data_q, user_ele_dict, user_num_dict)

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(units=188*2,input_shape=[188*3], activation='relu'),
    layers.Dense(units=188, activation='relu'),
    layers.Dense(units=1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

import math
import matplotlib.pyplot as plt
n = len(train_data_q)
batch_size = 256
train_loss = []
train_acc = []
valid_loss = []
valid_acc = []
epochs = 20

for epoch in range(epochs):
    epoch_train_loss = []
    epoch_train_acc = []
    for batch_index in range(math.ceil(n/batch_size)):
        
        if batch_index*batch_size+batch_size < n:
            X_train, y_train = pre_data(train_data_q[batch_index*batch_size:batch_index*batch_size+batch_size], user_ele_dict, user_num_dict)
        else:
            X_train, y_train = pre_data(train_data_q[batch_index*batch_size:], user_ele_dict, user_num_dict)
        
        batch_loss = model.train_on_batch(X_train, y_train,reset_metrics=False,return_dict=True)
        epoch_train_loss.append(batch_loss['loss'])
        epoch_train_acc.append(batch_loss['binary_accuracy'])
        
    train_loss.append(np.mean(epoch_train_loss))
    train_acc.append(np.mean(epoch_train_acc))

    epoch_valid_loss = model.test_on_batch(X_valid, y_valid,reset_metrics=False, return_dict=True)
    valid_loss.append(epoch_valid_loss['loss'])
    valid_acc.append(epoch_valid_loss['binary_accuracy'])
    print("Epoch: ", epoch, 
          "train_loss: ", round(np.mean(epoch_train_loss),4),
          "valid_loss: ", round(epoch_valid_loss['loss'],4),
          "train_acc: ", round(np.mean(epoch_train_acc),4),
          "valid_acc: ", round(epoch_valid_loss['binary_accuracy'],4))

model.save("my_h5_model.h5")
print("save completed!")
plt.plot(train_loss)
plt.plot(valid_loss)
plt.show()

plt.plot(train_acc)
plt.plot(valid_acc)
plt.show()

import pickle

with open('X.pkl', 'rb') as f:
    X = pickle.load(f)

with open('y.pkl', 'rb') as f:
    y = pickle.load(f)

print(X.shape)
print(y.shape)
print(X[0])
print(y)
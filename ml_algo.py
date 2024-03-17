from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import pickle

data = sns.load_dataset("diamonds")

X = data.drop('price', axis=1)
y = data['price']
encoder = LabelEncoder()
X['cut'] = encoder.fit_transform(X['cut'])
X['color'] = encoder.fit_transform(X['color'])
X['clarity'] = encoder.fit_transform(X['clarity'])

lr = LinearRegression()
lr.fit(X, y)

print(lr.score(X, y))

with open('model_lr.pkl', 'wb') as f:
    pickle.dump(lr, f)


rf = RandomForestRegressor()
rf.fit(X, y)

print(rf.score(X, y))

with open("model_rf.pkl", "wb") as f:
    pickle.dump(rf, f)

mlp = MLPRegressor(max_iter=500)
mlp.fit(X, y)

print(mlp.score(X, y))

with open("model_mlp.pkl", "wb") as f:
    pickle.dump(mlp, f)
import pandas as pd
from sklearn.model_selection import train_test_split

#load the data
dataset = pd.read_csv("data/raw/pima-indians-diabetes.csv")

# split the data into X an y
X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2020)
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)



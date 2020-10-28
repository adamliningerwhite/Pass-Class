import random
import joblib
import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Read data from file
path = "data.csv"
data = pd.read_csv(path, ",", error_bad_lines = False, warn_bad_lines = False)

# Passwords are features 
features = np.array(data['password'].astype('str'))

# Strength ratings are labels 
labels = np.array(data['strength'].astype('int'))

# Extract features from password strings 
vectorizer = TfidfVectorizer(analyzer='char') 
extracted_features = vectorizer.fit_transform(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(extracted_features, labels, test_size = 0.85, random_state = 47)

# Either load saved model or build from skratch
model = None
print("")
try:
    model = joblib.load('model.joblib')
    print("Model successfully loaded from file!", "\n")
except FileNotFoundError:
    print("Cannot find saved model, building one from scratch!", "\n")
    model = LogisticRegression(multi_class='multinomial', solver='sag')
    model.fit(train_features, train_labels)
    joblib.dump(model, 'model.joblib') # save model

# Show accuracy cuz we're very proud
print("Password classifier test accuracy: ", model.score(test_features, test_labels))
print("--------------------------------------------------------")

# Recieve user input and predict password strength using our model
while(True):
    response = input("Password: ")
    user_pass = [response]
    user_pass = vectorizer.transform(user_pass)
    pred = model.predict(user_pass)
    if pred[0] < 2:
        print("weak", "\n")
    else:
        print("strong", "\n")
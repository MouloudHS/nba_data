from flask import Flask, request, render_template, jsonify
import pickle

import pandas as pd
import numpy as np

from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin


import warnings
warnings.filterwarnings('ignore')
seed = 50

########################
def score_classifier(dataset, classifier, labels, scoring = "recall"):
    """
    Meme chose que MP Data sans le ré-entrainement et support 
    pour les dataframes et differentes methodes de scoring
    """
    kf = KFold(n_splits=3, shuffle=True)
    confusion_mat = np.zeros((2, 2))
    
    scores = []
    for train_idx, test_idx in kf.split(dataset):
        test_set = dataset.iloc[test_idx]
        test_labels = labels.iloc[test_idx]

        predicted_labels = classifier.predict(test_set)
        confusion_mat += confusion_matrix(test_labels, predicted_labels)
        if scoring=="recall":
            scores.append(recall_score(test_labels, predicted_labels))
        elif scoring=="f1":
            scores.append(f1_score(test_labels, predicted_labels))
        elif scoring=="accuracy":
            scores.append(accuracy_score(test_labels, predicted_labels))

    avg_score = np.mean(scores)
    print(confusion_mat)
    return avg_score

class NBAPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = pickle.load(open(model_path, 'rb')) 
        self.__threshold__ = -0.033131

    # Preprocessing :
    def __imputation__(self, df): 
        return df.dropna(axis=0)
    
    def __feature_engineering__(self, df):
        df = df.drop(["FGM", "FTM", "STL"], axis=1, inplace=False)
        return df

    def __preprocessing__(self, df):
        df = self.__imputation__(df)
        df = self.__feature_engineering__(df)
        return df

    def fit(self, X, y=None):
        return self
    
    # Prediction
    def predict(self, X):
        X = self.__preprocessing__(X)
        predictions = (self.model.decision_function(X) > self.__threshold__) * 1
        return predictions

def load_data(csv_data_file_path):
    csv_data = pd.read_csv(csv_data_file_path)    
    csv_data.dropna(inplace=True, ignore_index=True)
    csv_data = csv_data.sample(frac=0.2, replace=True, ignore_index=True)
    y_test = csv_data["TARGET_5Yrs"]
    csv_data = csv_data.drop(["Name", "TARGET_5Yrs"], axis=1)
    return csv_data, y_test
########################


# L'appli
app = Flask(__name__)
caracteristics = ["GP", "MIN", "PTS", "FGM", "FGA", "FG%", "3P Made", 
                  "3PA", "3P%", "FTM", "FTA", "FT%", "OREB", "DREB", 
                  "REB", "AST", "STL", "BLK", "TOV"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [ request.form.get(carac) for carac in caracteristics ] 
    
    # Initialization des caracs:
    player_data = pd.DataFrame([features], columns=caracteristics)
    
    # Predictions
    model_path = "adaBoost_best_model.sav"
    nba_predictor = NBAPredictor(model_path = model_path)
    prediction = nba_predictor.predict(player_data)

    # UN peu de Benchmarking (not a part of the test, but just for monitoring):
    print(f"{10*'--'} Quick Benchmark {10*'--'}")
    csv_data_file_path = "./Test_Data_Science/Test_Data_Science/nba_logreg.csv"
    csv_data, y_test = load_data(csv_data_file_path)

    for scoring_method in ["recall", "f1", "accuracy"]:
        print(f"Scoring Method = {scoring_method}")
        score = score_classifier(csv_data, nba_predictor, y_test, scoring = scoring_method)
        print(scoring_method + f": {score}")
        print()
            
    print(f"{30*'--'}")

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
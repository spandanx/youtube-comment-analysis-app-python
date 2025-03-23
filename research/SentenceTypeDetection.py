import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
import pickle
# import numpy as np
# import random

model_location = './Model/naivebayes_sentence_type_detection.pickle'

def predict_text(classifier, text):
    ans = classifier.classify(dialogue_act_features(text))
    return "question" if ans == 1 else "statement"

def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

def CreateSentenceDetectionModel():

    df_raw = pd.read_csv('../data/Sentence Types - Question, Command and Statement.csv')
    df = pd.DataFrame()
    df["statement"] = df_raw["statement"]
    df["type"] = df_raw["type"].apply(lambda typ: 1 if typ=="question" else 0)


    df["encoded"] = df["statement"].apply(dialogue_act_features)

    X_train, X_test, y_train, y_test = train_test_split(df["encoded"], df["type"], test_size = 0.25, random_state = 2024)

    train_set = [(X_train.iloc[i], y_train.iloc[i]) for i in range(len(X_train))]
    test_set = [(X_test.iloc[i], y_test.iloc[i]) for i in range(len(X_test))]

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print("Accuracy = ", nltk.classify.accuracy(classifier, test_set))

    save_classifier = open(model_location, "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()

def getClassifier():
    classifier_f = open(model_location, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier


def sentenceDetectionModel(sentenceArray, classifier):

    result = []
    for sentence in sentenceArray:
        # result.append("a - " + sentence)
        result.append({
            "text": sentence,
            "type": predict_text(classifier, sentence)
        })
    return result

if __name__ == "__main__":
    TestSentenceDetectionModel("would you like to tell me about that?")
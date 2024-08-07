import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
import pickle

from DataProcessing.Lemmatizer import Lemmatizer

# import numpy as np
# import random
data_path = '../data/Sentence Types - Question, Command and Statement.csv'
data_path_mod = '../data/Sentence Types - Question, Command and Statement - modified.csv'

model_location = '../Model/naivebayes_sentence_type_detection_test_pos_custom1.pickle'

class SentenceTypeDetection:

    def __init__(self):
        self.lemmatizer = Lemmatizer()

    def predict_text(self, classifier, text):
        ans = classifier.classify(self.dialogue_act_features(text))
        return "question" if ans == 1 else "statement"

    def dialogue_act_features(self, post):
        features = {}
        for word in nltk.word_tokenize(post):
            features['contains({})'.format(word.lower())] = True
        return features

    # def CreateSentenceDetectionModel(self):
    #
    #     df_raw = pd.read_csv(data_path)
    #     df = pd.DataFrame()
    #     df["statement"] = df_raw["statement"]
    #     df["type"] = df_raw["type"].apply(lambda typ: 1 if typ=="question" else 0)
    #
    #
    #     df["encoded"] = df["statement"].apply(self.dialogue_act_features)
    #
    #     X_train, X_test, y_train, y_test = train_test_split(df["encoded"], df["type"], test_size = 0.25, random_state = 2024)
    #
    #     train_set = [(X_train.iloc[i], y_train.iloc[i]) for i in range(len(X_train))]
    #     test_set = [(X_test.iloc[i], y_test.iloc[i]) for i in range(len(X_test))]
    #
    #     classifier = nltk.NaiveBayesClassifier.train(train_set)
    #     print("Accuracy = ", nltk.classify.accuracy(classifier, test_set))

        # save_classifier = open(model_location, "wb")
        # pickle.dump(classifier, save_classifier)
        # save_classifier.close()

    def getClassifier(self):
        classifier_f = open(model_location, "rb")
        classifier = pickle.load(classifier_f)
        classifier_f.close()
        return classifier


    def sentenceDetectionModel(self, sentenceArray, classifier):

        result = []
        for sentence in sentenceArray:
            # result.append("a - " + sentence)
            result.append({
                "text": sentence,
                "type": self.predict_text(classifier, sentence)
            })
        return result

    def lemmatized_word(self, sentence):
        lemmatized_words = self.lemmatizer.lemmatize(sentence)
        lemmatized_sentence = ' '.join([word["raw_word"] for word in lemmatized_words])
        return lemmatized_sentence

    def pos_word(self, sentence):
        lemmatized_words = self.lemmatizer.lemmatize(sentence)
        lemmatized_sentence = ' '.join([word["pos_tag"] for word in lemmatized_words])
        return lemmatized_sentence

    def custom_pos_word(self, sentence):
        lemmatized_words = self.lemmatizer.lemmatize(sentence)
        lemmatized_sentence = ' '.join([word["pos_tag"] if word["pos_tag"]!="SENDPUNC" else word["final_word"] for word in lemmatized_words])
        return lemmatized_sentence

    def CreateSentenceDetectionModel2(self):
        df_raw = pd.read_csv(data_path)
        # df_raw = df_raw.loc[:10]
        df = pd.DataFrame()
        df["statement"] = df_raw["statement"]
        df["type"] = df_raw["type"].apply(lambda typ: 1 if typ == "question" else 0)

        # for row in df['statement']:
        #     print(row)

        lemmatized_statements = df["statement"].apply(self.custom_pos_word)
        # for row in lemmatized_statements:
        #     print(row)

        df["encoded"] = lemmatized_statements.apply(self.dialogue_act_features)
        #
        X_train, X_test, y_train, y_test = train_test_split(df["encoded"], df["type"], test_size=0.25, random_state=2024)

        train_set = [(X_train.iloc[i], y_train.iloc[i]) for i in range(len(X_train))]
        test_set = [(X_test.iloc[i], y_test.iloc[i]) for i in range(len(X_test))]

        classifier = nltk.NaiveBayesClassifier.train(train_set)
        print("Accuracy = ", nltk.classify.accuracy(classifier, test_set))

        save_classifier = open(model_location, "wb")
        pickle.dump(classifier, save_classifier)
        save_classifier.close()

    def add_character(self, sentence, character):
        if sentence is None:
            return sentence
        if sentence[-1] not in ('.', '?', '!'):
            return sentence + character
        return sentence

    def get_end_char_freqs(self, df_column_series):
        freq_map = {}
        for row in df_column_series:
            if row[-1] in ('.', '?', '!'):
                if (row[-1] not in freq_map):
                    freq_map[row[-1]] = 0
                freq_map[row[-1]] = freq_map[row[-1]] + 1
        for key, val in freq_map.items():
            freq_map[key] = (val/len(df_column_series))
        return freq_map

    def modify_data(self):
        df_raw = pd.read_csv(data_path)
        df = pd.DataFrame()
        df["statement"] = df_raw["statement"]
        df["type"] = df_raw["type"]
        df_mod = df.sample(random_state=2024, frac=1, axis='rows')
        n = len(df_mod) * 0.75
        idx = 0
        while idx < len(df_mod) and n>0:
            xn = df_mod.iloc[[idx]]
            ix = xn.index[0]
            if (df_mod.loc[ix]['type'] == 'question'):
                sn = self.add_character(df_mod.loc[ix]['statement'], '?')
                # df_mod.loc[idx]['statement'] = sn
                df_mod.at[ix, 'statement'] = sn
                # df_mod.set_value('statement', idx, sn)
                n -= 1
            xy = df_mod.iloc[[idx]]
            idx += 1
        a = 'mn'
        # df_mod.to_csv(data_path_mod, index=False)
    def data_analysis(self, df_path):
        df = pd.read_csv(df_path)
        print("question")
        print(st_detection.get_end_char_freqs(df.loc[df['type'] == 'question']['statement']))
        print("statement")
        print(st_detection.get_end_char_freqs(df.loc[df['type'] == 'statement']['statement']))
        print("command")
        print(st_detection.get_end_char_freqs(df.loc[df['type'] == 'command']['statement']))

if __name__ == "__main__":
    st_detection = SentenceTypeDetection()
    st_detection.CreateSentenceDetectionModel2()
    # st_detection.data_analysis(data_path)

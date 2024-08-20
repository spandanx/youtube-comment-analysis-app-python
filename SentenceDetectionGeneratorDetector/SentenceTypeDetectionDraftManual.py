import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
import pickle

from DataProcessing.Lemmatizer import Lemmatizer

# import numpy as np
# import random
data_path = '../data/Sentence Types - Question, Command and Statement.csv'
data_path_mod = '../data/Sentence Types - Question, Command and Statement - modified.csv'

model_location = '../Model/naivebayes_sentence_type_detection_test_pos_custom1.pickle'

class SentenceTypeDetectionManual:

    def __init__(self):
        self.lemmatizer = Lemmatizer()
        self.pos_map = {
            "CC": "CC",
            "CD": "CD",
            "EX": "EX",
            "FW": "FW",
            "IN": "IN",
            "JJ": "JJ",
            "JJR": "JJ",
            "JJS": "JJ",
            "MD": "MD",
            "NNS": "NN",
            "NN": "NN",
            "NNP": "NN",
            "NNPS": "NN",
            "PRP": "PRP",
            "PRP$": "PRP",
            "PDT": "PDT",
            "POS": "POS",
            "RB": "RB",
            "RBR": "RB",
            "RBS": "RB",
            "RP": "RP",
            "SYM": "SYM",
            "TO": "TO",
            "UH": "UH",
            "VB": "V",
            "VBD": "V",
            "VBZ": "V",
            "VBN": "V",
            "VBG": "V",
            "VBP": "V",
            "WRB": "W",
            "WDT": "W",
            "WP": "W",
            "WP$": "W"
        }

    def pos_word(self, sentence):
        lemmatized_words = self.lemmatizer.lemmatize(sentence)
        lemmatized_sentence = [self.truncate_pos(word["pos_tag"]) for word in lemmatized_words]
        return lemmatized_sentence

    def predict_text(self, classifier, text):
        ans = classifier.classify(self.dialogue_act_features(text))
        return "question" if ans == 1 else "statement"

    def truncate_pos(self, pos):
        if pos in self.pos_map:
            return self.pos_map[pos]
        else:
            return pos

    def clean_spaces(self, sentence):
        return sentence.strip()

    def get_last_sentence(self, phrase):
        lst = re.split('\\.', phrase)
        if len(lst)==1:
            return self.clean_spaces(phrase)
        elif len(lst)>1:
            if (len(lst[-1]) == 0):
                return self.clean_spaces(lst[-2] + '.')
            else:
                return self.clean_spaces(lst[-1])

    def get_questions(self):
        # df_raw = pd.read_csv('../data/Sentence Types - Question, Command and Statement.csv')
        df_raw = pd.read_csv('../data/Sentence Types - Question, Command and Statement - modified.csv')
        df = pd.DataFrame()
        df["statement"] = df_raw["statement"]
        df["type"] = df_raw["type"]
        df_questions = df[df['type'] == "question"]
        questions = list(df_questions["statement"])
        return questions

    def get_questions_answers(self):
        # df_raw = pd.read_csv('../data/Sentence Types - Question, Command and Statement.csv')
        df_raw = pd.read_csv('../data/Sentence Types - Question, Command and Statement - modified.csv')
        df = pd.DataFrame()
        df["statement"] = df_raw["statement"]
        df["type"] = df_raw["type"]
        df_questions = df[(df['type'] == "question") | (df['type'] == "statement")]
        return df_questions

    def truncate_sentence_pos(self, sentence, min_length):
        last_sentence = self.get_last_sentence(sentence)
        truncated_sentence_pos = self.pos_word(last_sentence)
        # end_tag = "OTHER"
        # if last_sentence[-1] == ".":
        #     end_tag = "FULLSTOP"
        # elif last_sentence[-1] == "?":
        #     end_tag = "QUESTION"
        # pos_tag = truncated_sentence_pos[:min_length] + [end_tag]
        pos_tag = truncated_sentence_pos[:min_length]
        return pos_tag


if __name__ == "__main__":
    st_detection = SentenceTypeDetectionManual()
    # st_detection.data_analysis(data_path)
    st = ["What does it mean?", "What is your name?", "How do you know?", "Is it like this.",
          "Do you mean that.",
          "Are we all in the same page?",
          "Could you please do this?"]
    freq_2 = dict()
    freq_3 = dict()
    qs = st_detection.get_questions()
    pos_examples = dict()

    # phrase = "In 2012, the modern use of electronic educational technology (also called e-learning) had grown at 14 times the rate of traditional learning. What did surveys show in 2008?"
    phrase = "you arrive in  Bi..."
    print(st_detection.truncate_sentence_pos(phrase, 3))
    # phrase = " What did surveys show in 2008"
    # print(st_detection.clean_spaces(phrase))

    for sentence in qs[:150]:
        if len(sentence) != 0:
            # end_tag = "OTHER"
            # if sentence[-1] == ".":
            #     end_tag = "FULLSTOP"
            # elif sentence[-1] == "?":
            #     end_tag = "QUESTION"

            pos_tag = st_detection.pos_word(st_detection.get_last_sentence(sentence))
            # pos_2_tag = ' '.join(pos_tag[:2] + [end_tag])
            # pos_3_tag = ' '.join(pos_tag[:3] + [end_tag])
            pos_2_tag = ' '.join(st_detection.truncate_sentence_pos(sentence, 2))
            pos_3_tag = ' '.join(st_detection.truncate_sentence_pos(sentence, 3))
            if pos_2_tag not in freq_2:
                freq_2[pos_2_tag] = 1
            if pos_3_tag not in freq_3:
                freq_3[pos_3_tag] = 1
            freq_2[pos_2_tag] += 1
            freq_3[pos_3_tag] += 1
            # print(pos_2_tag, "\t, ", pos_3_tag, "\t -> ", sentence)

    print("2 KEY ------------------------")
    freq_2_keys = list(freq_2.keys())
    freq_2_keys.sort(key=lambda x:-freq_2[x])
    for key in freq_2_keys:
        print(key, freq_2[key])

    print("3 KEY ------------------------")
    freq_3_keys = list(freq_3.keys())
    freq_3_keys.sort(key=lambda x: -freq_3[x])
    for key in freq_3_keys:
        print(key, freq_3[key])


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing import sequence

from DataProcessing.SentenceDetectionGeneratorDetector.SentenceTypeDetectionDraftManual import SentenceTypeDetectionManual
import numpy as np
import pickle

model_location_100 = '../Model/lstm_model_209134_epoch_50.pickle'
model_location_10 = '../Model/lstm_model_1000_epoch_10.h5'
model_location_10_keras = '../Model/lstm_model_1000_epoch_10_keras.h5'
# model_location_full_keras = '../Model/lstm_model_209134_epoch_50_2.h5'
# model_location_full_keras_dropout = '../Model/lstm_model_209134_epoch_50_dropout.h5'
model_location_full_keras_dropout2 = '../Model/lstm_model_209134_epoch_50_dropout_without_end_tag.h5'
class LSTM_train:

    def __init__(self):
        self.s_detection_manual = SentenceTypeDetectionManual()

    def get_pos_tags(self, dataset, dataset_size):

        pos_tags = set()
        pos_tags.add("FULLSTOP")
        pos_tags.add("QUESTION")
        pos_tags.add("OTHER")

        for index, row in dataset.iloc[:dataset_size].iterrows():
            truncated_pos_sentence = self.s_detection_manual.pos_word(row["statement"])
            for ps in truncated_pos_sentence:
                pos_tags.add(ps)
        return pos_tags

    def convert_dataset_to_label_num_X_Y(self, pos_tags, dataset, min_len, dataset_size):
        pos_tag_list = sorted(list(pos_tags))
        self.pos_tag_map = {(pos_tag_list[i]): (i) for i in range(len(pos_tag_list))}
        self.pos_tag_reverse_map = {(i): (pos_tag_list[i]) for i in range(len(pos_tag_list))}
        self.target_label_map = {"statement": 0, "question": 1}
        self.target_label_reverse_map = {0: "statement", 1: "question"}

        pos_train_list = []
        target_list = []
        error_count = 0
        for index, row in dataset.iloc[0:dataset_size].iterrows():
            try:
                if len(row) > 0:
                    truncated_pos_sentence = self.s_detection_manual.truncate_sentence_pos(row["statement"], min_len)
                    pos_train_list.append([self.pos_tag_map[ps] for ps in truncated_pos_sentence])
                    target_list.append(self.target_label_map[row["type"]])
            except Exception as e:
                error_count += 1
                print("ERROR", error_count)
                print(row.statement)
                # print(row)
                print(index)
                print(e)
                # break
        print("Errors", error_count)
        return pos_train_list, target_list


        # for index, row in dataset.iloc[:dataset_size].iterrows():
        #     target_list.append(self.target_label_map[row["type"]])
        # return target_list

    # def convert_dataset_to_label_num_Y(self, dataset, dataset_size):
    #     self.target_label_map = {"statement": 0, "question": 1}
    #     self.target_label_reverse_map = {0: "statement", 1: "question"}
    #     target_list = []
    #     for index, row in dataset.iloc[:dataset_size].iterrows():
    #         target_list.append(self.target_label_map[row["type"]])
    #     return target_list

    def train_LSTM_model(self, X, y, min_len, epochs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)
        # (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
        # truncate and pad input sequences
        X_train = np.asarray(sequence.pad_sequences(X_train, maxlen=min_len))
        X_test = np.asarray(sequence.pad_sequences(X_test, maxlen=min_len))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        top_words = len(self.pos_tag_map)
        embedding_vecor_length = 32
        model = Sequential()
        model.add(Embedding(top_words, embedding_vecor_length, input_length=min_len))
        model.add(LSTM(100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(X_train, y_train, epochs=epochs, batch_size=64)
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        return model

    def train_LSTM_model_dropout(self, X, y, min_len, epochs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)
        # (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
        # truncate and pad input sequences
        X_train = np.asarray(sequence.pad_sequences(X_train, maxlen=min_len))
        X_test = np.asarray(sequence.pad_sequences(X_test, maxlen=min_len))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        top_words = len(self.pos_tag_map)
        embedding_vecor_length = 32
        model = Sequential()
        model.add(Embedding(top_words, embedding_vecor_length, input_length=min_len))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(X_train, y_train, epochs=epochs, batch_size=64)
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        return model

    def save_model(self, model, model_location):
        # save_classifier = open(model_location, "wb")
        # pickle.dump(model, save_classifier)
        # save_classifier.close()
        model.save(model_location)

    def load_model(self, model_location):
        classifier_f = open(model_location, "rb")
        model = pickle.load(classifier_f)
        classifier_f.close()
        return model

if __name__ == "__main__":
    lstm_train = LSTM_train()
    min_len = 3
    epochs = 50
    dataset_size = 209134
    # pd.options.display.max_colwidth = 100
    dataset = lstm_train.s_detection_manual.get_questions_answers()
    #209134
    # pos_tags = lstm_rough.get_pos_tags(dataset, dataset_size)
    pos_tags = set(['CD', 'DT', 'MD', 'QUESTION', 'NN', '(', 'UH', ')', "''", '``', 'PUNC', 'POS', 'TO', '#', 'V', 'PRP', 'SYM',
     'OTHER', 'IN', 'JJ', 'FULLSTOP', 'SENDPUNC', 'RP', '$', 'EX', 'CC', 'FW', 'RB', 'LS', 'PDT', 'W'])
    print(pos_tags)
    print("POS TAGS Generated")
    # dataset = pd.DataFrame([[" you arrive in  Bi...", "statement"], ["i'll give you thi...", "statement"]], columns=['statement', 'type'])

    X_dataset, y_dataset = lstm_train.convert_dataset_to_label_num_X_Y(pos_tags=pos_tags, dataset=dataset, min_len=min_len, dataset_size=dataset_size)
    # print("GOT X_dataset")
    # y_dataset = lstm_rough.convert_dataset_to_label_num_Y(dataset=dataset, dataset_size=dataset_size)
    print("GOT X, y dataset")
    model = lstm_train.train_LSTM_model_dropout(X_dataset, y_dataset, min_len, epochs)
    lstm_train.save_model(model, model_location_full_keras_dropout2)
    #with dropout = Accuracy: 99.91%
    #with dropout and no end tag = Accuracy: 91.89%
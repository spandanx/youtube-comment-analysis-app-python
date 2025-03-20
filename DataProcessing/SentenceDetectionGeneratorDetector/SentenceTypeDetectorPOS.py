from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from keras.models import load_model

from DataProcessing.SentenceDetectionGeneratorDetector.SentenceTypeDetectionDraftManual import SentenceTypeDetectionManual
import numpy as np
import pickle

# model_location_100 = '../Model/lstm_model_209134_epoch_50.h5'
# model_location_10 = '../Model/lstm_model_1000_epoch_10.h5'
# model_location_10_keras = '../Model/lstm_model_1000_epoch_10_keras.h5'
# model_location_full_keras = '../Model/lstm_model_209134_epoch_50_2.h5'
if __name__ == "__main__":
    model_location_full_keras_dropout2 = '../Model/lstm_model_209134_epoch_50_dropout_without_end_tag.h5'
else:
    model_location_full_keras_dropout2 = './Model/lstm_model_209134_epoch_50_dropout_without_end_tag.h5'

class SentenceTypeDetectorPOS:

    def __init__(self):
        self.s_detection_manual = SentenceTypeDetectionManual()
        self.lstm_model = self.load_model(model_location_full_keras_dropout2)
        self.target_label_map = {"statement": 0, "question": 1}
        self.target_label_reverse_map = {0: "statement", 1: "question"}
        self.pos_tags = set(['CD', 'DT', 'MD', 'QUESTION', 'NN', '(', 'UH', ')', "''", '``', 'PUNC', 'POS', 'TO', '#', 'V', 'PRP', 'SYM',
     'OTHER', 'IN', 'JJ', 'FULLSTOP', 'SENDPUNC', 'RP', '$', 'EX', 'CC', 'FW', 'RB', 'LS', 'PDT', 'W'])

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

        pos_train_list = []
        target_list = []
        error_count = 0
        for index, row in dataset.iloc[0:dataset_size].iterrows():
            try:
                if len(row) > 0:
                    truncated_pos_sentence = self.s_detection_manual.truncate_sentence_pos(row["statement"], min_len)
                    pos_train_list.append([self.pos_tag_map[ps] for ps in truncated_pos_sentence])
                    target_list.append(self.target_label_map[row["type"]])
            except:
                error_count += 1
                print("ERROR", error_count)
                print(row.statement)
                # print(row)
                print(index)
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

    def save_model(self, model, model_location):
        save_classifier = open(model_location, "wb")
        pickle.dump(model, save_classifier)
        save_classifier.close()

    def load_model(self, model_location):
        loaded_model = load_model(model_location)
        return loaded_model

    def convert_sentence(self, sentence, min_len):
        pos_tag_list = sorted(list(self.pos_tags))
        pos_tag_map = {(pos_tag_list[i]): (i) for i in range(len(pos_tag_list))}
        # print(pos_tag_map)
        truncated_pos_sentence = self.s_detection_manual.truncate_sentence_pos(sentence, min_len)
        return [pos_tag_map[ps] for ps in truncated_pos_sentence]

    def predict_sentence(self, sentence_vector):
        sentence_type_array = self.lstm_model.predict(x=sentence_vector)
        sentence_type_response = [self.target_label_reverse_map[(0 if score<=0.5 else 1)] for score in sentence_type_array]
        return sentence_type_response

    def predict_sentence_array(self, sentenceArray):
        result = []
        sentence_vector_array = [self.convert_sentence(sentence, 3) for sentence in sentenceArray]
        sentence_vector_array_filtered = list(filter(lambda sntce: len(sntce)>0, sentence_vector_array))
        sentence_type_array = self.predict_sentence(sentence_vector_array_filtered)
        for sentence, sentence_type in zip(sentenceArray, sentence_type_array):
            result.append({
                "text": sentence,
                "type": sentence_type
            })
        return result


if __name__ == "__main__":
    sentenceTypeDetectorPOS = SentenceTypeDetectorPOS()
    min_len = 3
    epochs = 50
    dataset_size = 209134
    # pd.options.display.max_colwidth = 100
    # dataset = lstm_rough.s_detection_manual.get_questions_answers()
    #209134
    # pos_tags = lstm_rough.get_pos_tags(dataset, dataset_size)
    # pos_tags = set(['CD', 'DT', 'MD', 'QUESTION', 'NN', '(', 'UH', ')', "''", '``', 'PUNC', 'POS', 'TO', '#', 'V', 'PRP', 'SYM',
    #  'OTHER', 'IN', 'JJ', 'FULLSTOP', 'SENDPUNC', 'RP', '$', 'EX', 'CC', 'FW', 'RB', 'LS', 'PDT', 'W'])
    # print(pos_tags)
    # print("POS TAGS Generated")
    # dataset = pd.DataFrame([[" you arrive in  Bi...", "statement"], ["i'll give you thi...", "statement"]], columns=['statement', 'type'])

    # X_dataset, y_dataset = lstm_rough.convert_dataset_to_label_num_X_Y(pos_tags=pos_tags, dataset=dataset, min_len=min_len, dataset_size=dataset_size)
    # print("GOT X_dataset")
    # y_dataset = lstm_rough.convert_dataset_to_label_num_Y(dataset=dataset, dataset_size=dataset_size)
    # model = lstm_rough.train_LSTM_model(X_dataset, y_dataset, min_len+1, epochs)
    sentence = "What year did Dell announce its plans to buy its building?"
    # sentence = "Apple tastes sweet"
    # model = lstm_load.load_model(model_location_full_keras_dropout2)
    # sentence_vector = lstm_load.convert_sentence(sentence, 3)
    sentence_type = sentenceTypeDetectorPOS.predict_sentence_array([sentence])
    print(sentence_type)
    # self.target_label_map = {"statement": 0, "question": 1}
    # [[0.9999994]]
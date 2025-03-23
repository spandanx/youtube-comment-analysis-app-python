from datasets import DatasetDict, Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding
import pandas as pd
import yaml
import inspect
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import pickle
from keras.models import load_model
from sklearn.metrics import accuracy_score

def get_caller_file_name():
    call_stack = inspect.stack()
    call_filenames = [stack.filename for stack in call_stack]
    common_file_name = "StockDocPython"
    filtered_filenames = [filename for filename in call_filenames if filename.endswith(".py") and common_file_name in filename]
    if len(filtered_filenames)==0:
        return "Not Found"
    caller_filename = os.path.basename(filtered_filenames[-1])
    return caller_filename

source_file_name = get_caller_file_name()
print("source_file_name", source_file_name)


if source_file_name == "LSTMSentenceTypeDetectionBertEmbeddingTrainer.py":
    dataset_file = "../../../data/Sentence Types - Question, Command and Statement.csv"
    model_export_path = "../../../Model/LSTM_BERT_Embedding/lstm_model_768_pre_padding.h5"
elif source_file_name == "YoutubeSearch.py":
    dataset_file = ".\\data\\Sentence Types - Question, Command and Statement.csv"
    model_export_path = ".\\Model\\LSTM_BERT_Embedding\\lstm_model_768_pre_padding.h5"
elif source_file_name == "main.py":
    dataset_file = ".\\data\\Sentence Types - Question, Command and Statement.csv"
    model_export_path = ".\\Model\\LSTM_BERT_Embedding\\lstm_model_768_pre_padding.h5"
else:
    dataset_file = "../../../data/Sentence Types - Question, Command and Statement.csv"
    model_export_path = "../../../Model/LSTM_BERT_Embedding/lstm_model_768_pre_padding.h5"

# dataset_file = "C:\\Users\\Spandan\\Downloads\\Compressed\\Sentence Types - Question, Command and Statement\\Sentence Types - Question, Command and Statement.csv"
# model_export_path = "D:\PROJECTS\TensorFlow Model Exports\LSTM Simple Question Detection\\lstm_model_768_pre_padding.h5"

class LSTMSentenceTypeDetectionBertEmbeddingTrainer:

    def __init__(self):
        self.lr = 2e-4
        self.batch_size = 8
        self.num_epochs = 5
        self.dataset_size = 1000
        self.vocabulary_size = 30522
        self.embedding_dimension = 768
        self.dataset_max_size = 1000
        self.embedding_model_path = "google-bert/bert-base-uncased"
        self.dataset_file = dataset_file
        self.model_export_path = model_export_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_path)


    def preprocess_function(self, examples):
        # return tokenized text with truncation
        return self.tokenizer(examples["text"], truncation=True)

    def tokenize_and_pad_train_target_data(self, dataset_dict):
        tokenized_data_raw = dataset_dict.map(self.preprocess_function, batched=True)
        df_train = tokenized_data_raw["train"].to_pandas()
        train_data = np.asarray(sequence.pad_sequences(df_train["input_ids"], maxlen=self.embedding_dimension,
                                                       padding='pre'))  # Post gives bad result
        target_data = np.asarray(df_train["labels"])
        return train_data, target_data


    def split_data(self, train_data, target_data):
        X_train, X_test, y_train, y_test = train_test_split(train_data, target_data, test_size=0.2, random_state=2024)
        return X_train, X_test, y_train, y_test

    def create_model(self):
        model = Sequential()
        model.add(Embedding(self.vocabulary_size, self.embedding_dimension, input_length=self.embedding_dimension))
        model.add(LSTM(100))  # Can be same as embedding_dimension, but model will large and inefficient
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def load_dataset(self):
        df_raw = pd.read_csv(self.dataset_file)
        df_raw.columns = ["text", "labels"]
        df = df_raw.loc[df_raw["labels"] != "command"]
        df["labels"] = df["labels"].map(lambda typ: 0 if typ == "statement" else 1)
        question_df = df.loc[df["labels"] == 1]
        statement_df = df.loc[df["labels"] == 0]
        min_size = min(len(question_df), len(statement_df))
        question_df = question_df.iloc[:min_size]
        statement_df = statement_df.iloc[:min_size]

        merged_df = pd.concat([question_df, statement_df], axis=0)
        merged_df = merged_df.sample(frac=1).reset_index(drop=True)

        df_small = merged_df.iloc[:self.dataset_max_size]
        df_small = df_small.reset_index(drop=True)
        dataset = Dataset.from_pandas(df_small).train_test_split(test_size=0.10)
        dataset_dict = DatasetDict({"train": dataset["train"], "validation": dataset["test"]})

        print(dataset_dict["train"].to_pandas().labels.value_counts())
        return dataset_dict


    def train_model(self, model, X_train, y_train):
        model.fit(X_train, y_train, epochs=self.num_epochs, batch_size=self.batch_size)

    def save_model(self, model):
        model.save_weights(self.model_export_path)

    def load_model(self):
        loaded_model = self.create_model()
        loaded_model.load_weights(self.model_export_path)
        return loaded_model

    def tokenizer_text_df(self, text_array):
        # return tokenized text with truncation
        tokenized_data = self.tokenizer(text_array, truncation=True)["input_ids"]
        return tokenized_data


    def tokenize_and_pad_dataframe_prediction(self, text_dataset):
        tokenized_data = text_dataset.apply(self.tokenizer_text_df)
        padded_encoding_test = sequence.pad_sequences(tokenized_data, maxlen=self.embedding_dimension, padding='pre')
        return padded_encoding_test

    def predict_sentence(self, sentence_array):
        df = pd.DataFrame(sentence_array, columns = ["text"])
        padded_data = self.tokenize_and_pad_dataframe_prediction(df["text"])
        loaded_model = self.load_model()
        pred = loaded_model.predict(padded_data)
        sentence_type_response = [0 if score <= 0.5 else 1 for score in pred]
        return sentence_type_response


if __name__ == "__main__":
    lstm = LSTMSentenceTypeDetectionBertEmbeddingTrainer()
    ############ Training ###########
    # dataset = lstm.load_dataset()
    # train_data, target_data = lstm.tokenize_and_pad_train_target_data(dataset)
    # X_train, X_test, y_train, y_test = lstm.split_data(train_data, target_data)
    # model = lstm.create_model()
    # lstm.train_model(model, X_train, y_train)
    # lstm.save_model(model)
    ############ Training ###########
    ############ Prediction ###########
    text_array = [
        "What developed from the mammalian odor pathways?",
        "Could you please tell me the direction of the retaurant?",
        "How do you know this?",
        "Burke received a vote of thanks from the Commons for his services in the Hastings Trial and he immediately resigned his seat"
    ]
    pred = lstm.predict_sentence(text_array)
    print(pred)
    ############ Prediction ###########

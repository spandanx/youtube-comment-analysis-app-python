import keras.models
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import json

#dataset : https://www.kaggle.com/datasets/warcoder/transliteration-dataset-21-indic-languages?resource=download
data_path = "C:/Users/spand/Downloads/Compressed/Transliteration Dataset 21 Indic languages/data/ben/ben_test.json"
model_path = '../../Model/sample_transliteration2.keras'

class Transliteration:

    def __init__(self, batch_size, epochs, latent_dim, num_samples):
        self.batch_size = batch_size
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.num_samples = num_samples

    def read_data(self):
        df = pd.read_json(data_path, lines=True)
        print(df.head())

    def vectorize_data(self):
        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()
        input_characters.add(' ')
        target_characters.add(' ')
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(num_samples, len(lines) - 1)]:
            each_line = json.loads(line)
            # print(type(line))
            # print(each_line)
            # print(type(each_line))
            input_text, target_text = each_line["english word"], each_line["native word"]  # we use -tab- os the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)
        self.input_characters = input_characters
        self.target_characters = target_characters
        self.input_texts = input_texts
        self.target_texts = target_texts

    def generate_tokens(self):
        self.input_characters = sorted(list(self.input_characters))
        self.target_characters = sorted(list(self.target_characters))
        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])

        self.input_token_index = dict([(char, i) for i, char in enumerate(self.input_characters)])
        self.target_token_index = dict([(char, i) for i, char in enumerate(self.target_characters)])
        self.reverse_target_char_index = dict([(i, char) for i, char in enumerate(self.target_characters)])

    # def generate_encoder_decoder_data(self):
    #     self.encoder_input_data = np.zeros((len(input_texts), self.max_encoder_seq_length, self.num_encoder_tokens), dtype='float32')
    #     self.decoder_input_data = np.zeros((len(input_texts), self.max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')
    #     self.decoder_target_data = np.zeros((len(input_texts), self.max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')

    def generate_encoder_sequence(self, input_texts):
        encoder_input_data = np.zeros((len(input_texts), self.max_encoder_seq_length, self.num_encoder_tokens),
                                      dtype='float32')
        for i, (input_text) in enumerate(input_texts):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.
            encoder_input_data[i, t + 1:, self.input_token_index[' ']] = 1.
        return encoder_input_data
    def generate_sequence_encoder_decoder(self):

        # encoder_input_data = np.zeros((len(self.input_texts), self.max_encoder_seq_length, self.num_encoder_tokens),
        #                                    dtype='float32')
        decoder_input_data = np.zeros((len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
                                           dtype='float32')
        decoder_target_data = np.zeros((len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
                                            dtype='float32')

        # for i, (input_text) in enumerate(self.input_texts):
        #     for t, char in enumerate(input_text):
        #         encoder_input_data[i, t, self.input_token_index[char]] = 1.
        #     encoder_input_data[i, t + 1:, self.input_token_index[' ']] = 1.
        encoder_input_data = self.generate_encoder_sequence(self.input_texts)

        for i, (target_text) in enumerate(self.target_texts):
            for t, char in enumerate(target_text):
                # decoder_target data is ahead of decoder input_dota by one timestep
                decoder_input_data[i, t, self.target_token_index[char]] = 1.
                if t > 0:
                    # decoder target data wiLL be ahead by one timestep
                    # and wiLL not include the start character.
                    decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.
            decoder_input_data[i, t + 1:, self.target_token_index[' ']] = 1.
            decoder_target_data[i, t:, self.target_token_index[' ']] = 1.

        self.encoder_input_data = encoder_input_data
        self.decoder_input_data = decoder_input_data
        self.decoder_target_data = decoder_target_data
        # self.target_token_index = target_token_index
    def create_encoder_model(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.encoder_inputs = encoder_inputs
        self.encoder_states = encoder_states
        self.decoder_inputs = decoder_inputs
        self.decoder_outputs = decoder_outputs
        self.decoder_dense = decoder_dense
        self.decoder_lstm = decoder_lstm
        self.model = model

        print(self.encoder_input_data.shape)
        print(self.decoder_input_data.shape)
        print(self.decoder_target_data.shape)
        self.model.summary()

    def fit_model(self):
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.2)

    def save_model(self, model):
        model.save(model_path)

    def create_decoder_model(self):
        encoder_model = Model(self.encoder_inputs, self.encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = self.decoder_lstm(
            self.decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)
        decoder_model = Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def decode_sequence(self, input_word):
        # Encode the input as state vectors.
        input_seq = self.generate_encoder_sequence([input_word])
        # keras.models.load_model()
        # unpickled_model = keras.models.load_model(model_path)

        states_value = self.encoder_model.predict(input_seq)
        # states_value = unpickled_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def train_model(self):
        pass

    def test_model(self):
        pass

    def detect_transliteration_from_text(self):
        pass

if __name__ == "__main__":
    batch_size = 64
    epochs = 100
    latent_dim = 256
    num_samples = 10000

    tl = Transliteration(batch_size, epochs, latent_dim, num_samples)
    # transliteration.read_data()
    tl.vectorize_data()
    tl.generate_tokens()
    tl.generate_sequence_encoder_decoder()
    tl.create_encoder_model()
    tl.fit_model()
    tl.create_decoder_model()
    tl.decode_sequence("karis")
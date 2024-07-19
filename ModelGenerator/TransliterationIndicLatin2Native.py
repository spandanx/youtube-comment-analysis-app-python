import keras.models
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import json
import pickle

#dataset : https://www.kaggle.com/datasets/warcoder/transliteration-dataset-21-indic-languages?resource=download


class TransliterationIndicLatin2Native:

    def __init__(self, batch_size=64, epochs=100, latent_dim=256, num_samples=10000):
        self.langs = ["ben", "guj", "hin", "kan", "mal", "mar", "nep", "pan", "ori", "san", "tam", "tel", "urd"]
        self.batch_size = batch_size
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.num_samples = num_samples

        self.input_texts = dict()
        self.target_texts = dict()
        self.input_characters = dict()
        self.target_characters = dict()
        self.num_encoder_tokens = dict()
        self.num_decoder_tokens = dict()
        self.max_encoder_seq_length = dict()
        self.max_decoder_seq_length = dict()
        self.input_token_index = dict()
        self.target_token_index = dict()
        self.reverse_target_char_index = dict()
        self.encoder_input_data = dict()
        self.decoder_input_data = dict()
        self.decoder_target_data = dict()
        self.encoder_inputs = dict()
        self.encoder_states = dict()
        self.decoder_inputs = dict()
        self.decoder_outputs = dict()
        self.decoder_dense = dict()
        self.decoder_lstm = dict()
        self.model = dict()
        self.encoder_model = dict()
        self.decoder_model = dict()

        self.encoding_model_path_base = '../Model/translit_encode/transliteration_{lang}_latin_to_{lang}_native_v2_encode.keras'
        self.decoding_model_path_base = '../Model/translit_decode/transliteration_{lang}_latin_to_{lang}_native_v2_decode.keras'
        self.decoding_token_desc_path_base = '../Model/decode_desc/decode_desc_{lang}_latin_to_{lang}_native_v2.keras'

        # self.vectorize_data()
        # self.generate_tokens()

    def vectorize_data(self, data_path, lang):
        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()
        input_characters.add(' ')
        target_characters.add(' ')
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(self.num_samples, len(lines) - 1)]:
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
        self.input_characters[lang] = input_characters
        self.target_characters[lang] = target_characters
        self.input_texts[lang] = input_texts
        self.target_texts[lang] = target_texts

    def generate_tokens(self, lang, decode_desc_path):
        self.input_characters[lang] = sorted(list(self.input_characters[lang]))
        self.target_characters[lang] = sorted(list(self.target_characters[lang]))
        self.num_encoder_tokens[lang] = len(self.input_characters[lang])
        self.num_decoder_tokens[lang] = len(self.target_characters[lang])
        self.max_encoder_seq_length[lang] = max([len(txt) for txt in self.input_texts[lang]])
        self.max_decoder_seq_length[lang] = max([len(txt) for txt in self.target_texts[lang]])

        self.input_token_index[lang] = dict([(char, i) for i, char in enumerate(self.input_characters[lang])])
        self.target_token_index[lang] = dict([(char, i) for i, char in enumerate(self.target_characters[lang])])
        self.reverse_target_char_index[lang] = dict([(i, char) for i, char in enumerate(self.target_characters[lang])])

        decode_description = {
            "num_decoder_tokens": self.num_decoder_tokens[lang],
            "reverse_target_char_index": self.reverse_target_char_index[lang],
            "max_decoder_seq_length": self.max_decoder_seq_length[lang],
            "max_encoder_seq_length": self.max_encoder_seq_length[lang],
            "num_encoder_tokens": self.num_encoder_tokens[lang],
            "input_token_index": self.input_token_index[lang]
        }
        self.write_to_file(decode_description, decode_desc_path)
        # num_decoder_tokens
        # reverse_target_char_index
        # max_decoder_seq_length

    def write_to_file(self, data, decode_desc_path):
        with open(decode_desc_path, 'wb') as f:
            pickle.dump(data, f)

    def load_char_maps(self, decode_desc_path):
        with open(decode_desc_path, 'rb') as f:
            data = pickle.load(f)
            return data

    def generate_one_hot_sequence(self, input_texts, lang):
        # max_encoder_seq_length
        # num_encoder_tokens
        # input_token_index
        encoder_input_data = np.zeros((len(input_texts), self.max_encoder_seq_length[lang], self.num_encoder_tokens[lang]),
                                      dtype='float32')
        for i, (input_text) in enumerate(input_texts):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self.input_token_index[lang][char]] = 1.
            encoder_input_data[i, t + 1:, self.input_token_index[lang][' ']] = 1.
        return encoder_input_data

    def generate_one_hot_encoder_decoder(self, lang):

        decoder_input_data = np.zeros((len(self.input_texts[lang]), self.max_decoder_seq_length[lang], self.num_decoder_tokens[lang]),
                                           dtype='float32')
        decoder_target_data = np.zeros((len(self.input_texts[lang]), self.max_decoder_seq_length[lang], self.num_decoder_tokens[lang]),
                                            dtype='float32')

        encoder_input_data = self.generate_one_hot_sequence(self.input_texts[lang], lang)

        for i, (target_text) in enumerate(self.target_texts[lang]):
            for t, char in enumerate(target_text):
                # decoder_target data is ahead of decoder input_dota by one timestep
                decoder_input_data[i, t, self.target_token_index[lang][char]] = 1.
                if t > 0:
                    # decoder target data wiLL be ahead by one timestep
                    # and wiLL not include the start character.
                    decoder_target_data[i, t - 1, self.target_token_index[lang][char]] = 1.
            decoder_input_data[i, t + 1:, self.target_token_index[lang][' ']] = 1.
            decoder_target_data[i, t:, self.target_token_index[lang][' ']] = 1.

        self.encoder_input_data[lang] = encoder_input_data
        self.decoder_input_data[lang] = decoder_input_data
        self.decoder_target_data[lang] = decoder_target_data
        # self.target_token_index = target_token_index

    def create_encoder_model(self, lang):
        # Define an input sequence and process it.
        encoder_inputs= Input(shape=(self.max_encoder_seq_length[lang], self.num_encoder_tokens[lang]), name='encoder_inputs')

        masking_array = [0 for _ in range(self.max_encoder_seq_length[lang])]
        masking_array[0] = 1

        masking = keras.layers.Masking(mask_value= masking_array)
        encoder_inputs_masked = masking(encoder_inputs)

        encoder_lstm=LSTM(latent_dim, return_state=True, name='encoder_lstm')
        LSTM_outputs, state_h, state_c = encoder_lstm(encoder_inputs_masked)

        # We discard `LSTM_outputs` and only keep the other states.
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, self.num_decoder_tokens[lang]), name='decoder_inputs')
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')

        # Set up the decoder, using `context vector` as initial state.
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)

        #complete the decoder model by adding a Dense layer with Softmax activation function
        #for prediction of the next output
        #Dense layer will output one-hot encoded representation as we did for input
        #Therefore, we will use input_dimension number of neurons
        decoder_dense = Dense(self.num_decoder_tokens[lang], activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        # put together
        model_encoder_training = Model([encoder_inputs, decoder_inputs],
                                       decoder_outputs, name='model_encoder_training')

        self.encoder_inputs[lang] = encoder_inputs
        self.encoder_states[lang] = encoder_states
        self.decoder_inputs[lang] = decoder_inputs
        self.decoder_outputs[lang] = decoder_outputs
        self.decoder_dense[lang] = decoder_dense
        self.decoder_lstm[lang] = decoder_lstm
        self.model[lang] = model_encoder_training

        print(self.encoder_input_data[lang].shape)
        print(self.decoder_input_data[lang].shape)
        print(self.decoder_target_data[lang].shape)
        self.model[lang].summary()

    def fit_model(self, lang):
        self.model[lang].compile(optimizer='adam', loss='categorical_crossentropy',
                                       metrics=['accuracy'])

        self.model[lang].fit([self.encoder_input_data[lang], self.decoder_input_data[lang]], self.decoder_target_data[lang],
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_split=0.2)

    def save_model(self, model, model_path):
        model.save(model_path)

    def load_encoding_decoding_models(self, encoding_model_path, decoding_model_path, lang, decode_desc_path):
        decode_desc = self.load_char_maps(decode_desc_path)
        self.num_decoder_tokens[lang] = decode_desc["num_decoder_tokens"]
        self.reverse_target_char_index[lang] = decode_desc["reverse_target_char_index"]
        self.max_decoder_seq_length[lang] = decode_desc["max_decoder_seq_length"]
        self.max_encoder_seq_length[lang] = decode_desc["max_encoder_seq_length"]
        self.num_encoder_tokens[lang] = decode_desc["num_encoder_tokens"]
        self.input_token_index[lang] = decode_desc["input_token_index"]

        self.encoder_model[lang] = keras.models.load_model(encoding_model_path)
        self.decoder_model[lang] = keras.models.load_model(decoding_model_path)

    def define_encoder_decoder_model(self, lang):
        encoder_model = Model(self.encoder_inputs[lang], self.encoder_states[lang])

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = self.decoder_lstm[lang](
            self.decoder_inputs[lang], initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense[lang](decoder_outputs)
        decoder_model = Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        self.encoder_model[lang] = encoder_model
        self.decoder_model[lang] = decoder_model


    def decode_sequence(self, input_word, lang):
        # Encode the input as state vectors.
        input_seq = self.generate_one_hot_sequence([input_word], lang)

        states_value = self.encoder_model[lang].predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens[lang]))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, 0] = 1
        # START (0 zero) in one-hot-encoding --> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        # decoded_seq = list()
        decoded_word = ''

        while not stop_condition:

            # in a loop
            # decode the input to a token/output prediction + required states for context vector
            output_tokens, h, c = self.decoder_model[lang].predict(
                [target_seq] + states_value)

            # convert the token/output prediction to a token/output
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            # sampled_char = sampled_token_index
            sampled_char = self.reverse_target_char_index[lang][sampled_token_index]
            decoded_word += sampled_char
            # add the predicted token/output to output sequence
            # decoded_seq.append(sampled_char)

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_token_index == 0 or sampled_token_index == 1 or
                    len(decoded_word) == self.max_decoder_seq_length[lang]):
                stop_condition = True

            # Update the input target sequence (of length 1)
            # with the predicted token/output
            target_seq = np.zeros((1, 1, self.num_decoder_tokens[lang]))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update input states (context vector)
            # with the output states
            states_value = [h, c]

            # loop back.....

        # when loop exists return the output sequence
        print(decoded_word)
        return decoded_word

    def test_model(self):
        pass

    def detect_transliteration_from_text(self):
        pass

    def transliterate_to_native(self, word, lang):
        self.load_encoding_decoding_models(self.encoding_model_path_base.format(lang=lang), self.decoding_model_path_base.format(lang=lang), lang, self.decoding_token_desc_path_base.format(lang=lang))
        return self.decode_sequence(word, lang).replace('\n', '')

    def train_model(self, lang, data_path, encoding_model_path, decoding_model_path, decode_desc_path):
        self.vectorize_data(data_path, lang)
        self.generate_tokens(lang, decode_desc_path)
        self.generate_one_hot_encoder_decoder(lang)
        self.create_encoder_model(lang)
        self.fit_model(lang)
        self.define_encoder_decoder_model(lang)
        encoder_model = self.encoder_model[lang]
        decoder_model = self.decoder_model[lang]
        self.save_model(encoder_model, encoding_model_path)
        self.save_model(decoder_model, decoding_model_path)

if __name__ == "__main__":
    batch_size = 64
    epochs = 100
    latent_dim = 256
    num_samples = 10000

    input_data_path_base = "C:/Users/spand/Downloads/Compressed/Transliteration Dataset 21 Indic languages/data/{lang}/{lang}_test.json"
    # model_path_base = '../Model/transliteration_{lang}_latin_to_{lang}_native_v2.keras'
    encoding_model_path_base = '../Model/translit_encode/transliteration_{lang}_latin_to_{lang}_native_v2_encode.keras'
    decoding_model_path_base = '../Model/translit_decode/transliteration_{lang}_latin_to_{lang}_native_v2_decode.keras'
    decoding_token_desc_path_base = '../Model/decode_desc/decode_desc_{lang}_latin_to_{lang}_native_v2.keras'

    tl = TransliterationIndicLatin2Native(batch_size, epochs, latent_dim, num_samples)
    langs = ['ben']
    # for lang in langs:
    #     input_data_path = input_data_path_base.format(lang=lang)
    #     encoding_model_path = encoding_model_path_base.format(lang=lang)
    #     decoding_model_path = decoding_model_path_base.format(lang=lang)
    #     decoding_token_desc_path = decoding_token_desc_path_base.format(lang=lang)
    #     tl.train_model(lang, input_data_path, encoding_model_path, decoding_model_path, decoding_token_desc_path)
    # decoding_token_desc_path = decoding_token_desc_path_base.format(lang=lang)
    tl.transliterate_to_native("jadukori", "ben")

    # transliteration.read_data()

    # tl.vectorize_data()
    # tl.generate_tokens()
    # tl.generate_one_hot_encoder_decoder()
    # tl.create_encoder_model()
    # tl.fit_model()
    # tl.define_encoder_decoder_model()
    # tl.save_model(tl.encoder_model, encoding_model_path)
    # tl.save_model(tl.decoder_model, decoding_model_path)

    # tl.load_encoding_decoding_models(encoding_model_path, decoding_model_path)
    # tl.decode_sequence("karis")
    # tl.decode_sequence("jadukori")
    # tl.transliterate_to_native("jadukori")
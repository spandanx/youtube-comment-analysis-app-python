from nltk.corpus import words

if __name__ == "__main__":
    nltk_words = set(words.words())
    file_name = '../../data/english_word_set/english_word_set.txt'
    with open(file_name, 'w') as f:
        f.write(str(nltk_words))
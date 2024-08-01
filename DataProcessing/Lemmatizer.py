import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
import pandas as pd

class Lemmatizer:

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.pos_key_map = {
        "CC": "coordinating conjunction",
        "CD": "cardinal digit",
        "DT": "determiner",
        "EX": "existential there",
        "FW": "foreign word",
        "IN": "preposition/subordinating conjunction",
        "JJ": "This NLTK POS Tag is an adjective (large)",
        "JJR": "adjective, comparative (larger)",
        "JJS": "adjective, superlative (largest)",
        "LS": "list market",
        "MD": "modal (could, will)",
        "NN": "noun, singular (cat, tree)",
        "NNS": "noun plural (desks)",
        "NNP": "proper noun, singular (sarah)",
        "NNPS": "proper noun, plural (indians or americans)",
        "PDT": "predeterminer (all, both, half)",
        "POS": "possessive ending (parent\ ‘s)",
        "PRP": "personal pronoun (hers, herself, him, himself)",
        "PRP$": "possessive pronoun (her, his, mine, my, our )",
        "PUNC": "punctuation (, : ;)",
        "RB": "adverb (occasionally, swiftly)",
        "RBR": "adverb, comparative (greater)",
        "RBS": "adverb, superlative (biggest)",
        "RP": "particle (about)",
        "SENDPUNC": "sentence end punctuation(. ! ?)",
        "TO": "infinite marker (to)",
        "UH": "interjection (goodbye)",
        "VB": "verb (ask)",
        "VBG": "verb gerund (judging)",
        "VBD": "verb past tense (pleaded)",
        "VBN": "verb past participle (reunified)",
        "VBP": "verb, present tense not 3rd person singular(wrap)",
        "VBZ": "verb, present tense with 3rd person singular (bases)",
        "WDT": "wh-determiner (that, what)",
        "WP": "wh- pronoun (who)",
        "WRB": "wh- adverb (how)"
    }
    def pos_tagger(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def lemmatize(self, sentence):
        pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence.lower()))
        lemmatized_sentence = []
        for word, tag in pos_tagged:
            if tag in [',', ':']:
                tag = "PUNC"
            elif tag in ['.']:
                tag = "SENDPUNC"
            _tag = self.pos_tagger(tag)
            word_details = {
                "raw_word": word,
                "lemma_tag": _tag,
                "pos_tag": tag,
                "pos_tag_details": self.pos_key_map[tag] if tag in self.pos_key_map else "NOT_FOUND"
            }
            if _tag is not None:
                word_details.update({"final_word": self.lemmatizer.lemmatize(word, self.pos_tagger(tag))})
            else:
                word_details.update({"final_word": word})
            lemmatized_sentence.append(word_details)
            # print(self.pos_tagger(tag))
        return lemmatized_sentence

    def lemmatize_test(self, sentence, pos_map):
        pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
        for word, tag in pos_tagged:
            if tag not in pos_map:
                pos_map[tag] = set()
            pos_map[tag].add(word)

if __name__ == "__main__":
    lemmatizer = Lemmatizer()

    # df = pd.read_csv('C:/users/spand/Downloads/youtoxic_english_1000.csv')
    # pos_map = {}
    # for index, row in df.iterrows():
    #     lemmatizer.lemmatize_test(row["Text"].lower(), pos_map)
    #     break
        # print(row["Text"])
    # print(pos_map)
    # print(pos_map.keys())
    # for key, val in pos_map.items():
    #     if key in pos_key_map:
    #         print(pos_key_map[key] + "(" + key + "):", "\n", "\t", val)
    #     else:
    #         print(key + ":", "\n", "\t", val)
    # print(dir(wordnet))
    # sentence = 'consulting'
    sentence = "While consulting, ; : I sometimes tell people about the consulting business. ? !"
    # sentence = "What would this be?"
    res = lemmatizer.lemmatize(sentence)
    for w in res:
        print(w)
    # print(wordnet.nomap)
    # print(wordnet.ADJ_SAT)
    # print("---------------")
    # print(wordnet.NOUN)
    # print(wordnet.ADV)
    # print(wordnet.VERB)
    # print(wordnet.ADJ)
    # print("---------------")
    # print(wordnet.lg_attrs)
    # print(wordnet.map30)
    # print(wordnet.MORPHOLOGICAL_SUBSTITUTIONS)
    # print(wordnet.omw_langs)
    # print(wordnet.provenances)
    # print("---------------")
    # print(wordnet.root)
    # print(wordnet.splits)
    # print(wordnet.tup)
    # abc = lemmatizer.lemmatizer.lemmatize('what', lemmatizer.pos_tagger('wh'))
    # print(res)
    # lemmatizer = WordNetLemmatizer()
    # sentence = 'consulting'
    # # test = nltk.word_tokenize(sentence)
    # pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    # lemmatized_sentence = []
    # for word, tag in pos_tagged:
    #     lemmatized_sentence.append(lemmatizer.lemmatize(word, pos_tagger(tag)))
    # print(lemmatized_sentence)
# Output: ['consult']
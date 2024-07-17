from time import time
import enchant

from LanguageDetector.OtherLanguageCharacterDetector import OtherLanguageCharacterDetector

d = enchant.Dict("en_US")
# import nltk
# nltk.download()
from nltk.corpus import words
nltk_words = set(words.words())

otherLanguageCharacterDetector = OtherLanguageCharacterDetector()

text_path = "C:\\Users\\spand\\OneDrive\\Documents\\Sample_english_text.txt"

file = open(text_path, "r")
content = file.read()
# print(content)
file.close()

lines = content.split('\n')

t = time()
nltk_wordset = set()
enchant_wordset = set()

lines = [
    "are more simply constructed from the 7-orthoplex.",
    "are farmers, while an additional 5% receives their livelihood from raising livestock.",
    "বসন্তের, ভ্রমণ, निर्माली",
    "nirmaali shivaalapurva siddhanto"
]

for line in lines:
    print(line)
    stringlst = line.split(' ')
    for string in stringlst:
        if len(string)==0:
            continue

        if d.check(string):
            print("Present - ", string)
            # enchant_wordset.add(string)
        else:
            print("Not present - ", string)
            print(otherLanguageCharacterDetector.detect_word_lang(string))


        # if string in nltk_words:
        #     nltk_wordset.add(string)

t1 = time()
print(t1 - t)
# for wd in enchant_wordset:
#     if wd not in nltk_wordset:
#         print(wd)
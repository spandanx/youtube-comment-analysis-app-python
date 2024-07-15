from time import time
import enchant
d = enchant.Dict("en_US")
# import nltk
# nltk.download()
from nltk.corpus import words
nltk_words = set(words.words())

text_path = "C:\\Users\\spand\\OneDrive\\Documents\\Sample_english_text.txt"

file = open(text_path, "r")
content = file.read()
# print(content)
file.close()

lines = content.split('\n')

t = time()
nltk_wordset = set()
enchant_wordset = set()

for line in lines:
    stringlst = line.split(' ')
    for string in stringlst:
        if len(string)==0:
            continue

        if d.check(string):
            enchant_wordset.add(string)

        if string in nltk_words:
            nltk_wordset.add(string)

t1 = time()
print(t1 - t)
for wd in enchant_wordset:
    if wd not in nltk_wordset:
        print(wd)
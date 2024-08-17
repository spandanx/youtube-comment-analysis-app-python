from nltk.tokenize import sent_tokenize
my_text = "A first affirmation is that Python is useful. But can Python be useful to me?"
sentences = sent_tokenize(my_text)
print(sentences)
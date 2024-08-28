# pip install transformers
import torch

from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)

text = """
There are two ways of extracting text using TextRank: keyword and sentence extraction. 
Keyword extraction can be done by simply using a frequency test, but this would almost always prove to be inaccurate. This is where TextRank automates the process to semantically provide far more accurate results based on the corpus.
Sentence extraction, on the other hand, studies the corpus to summarize the most valid sentences pertaining to the subject matter and phonologically arranges it. 
"""

#############
inputs = tokenizer.encode("summarize: " + text,
return_tensors='pt',
max_length=512,
truncation=True)

#############
summary_ids = model.generate(inputs, max_length=150, min_length=80, length_penalty=5., num_beams=2)

#############
summary = tokenizer.decode(summary_ids[0])

print(summary)

# from gensim.summarization.summarizer import summarize
from gensim.summarization import summarize
# from gensim.summarization import keywords
import wikipedia
import en_core_web_sm

wikisearch = wikipedia.page("https://en.wikipedia.org/wiki/India_Gate")
wikicontent = wikisearch.content
nlp = en_core_web_sm.load()
doc = nlp(wikicontent)

summ_per = summarize(wikicontent, ratio=0.01)
print("Percent summary")
print(summ_per)

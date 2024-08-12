import nltk
import pandas as pd
nltk.download('nps_chat')
#len - 10567
# posts = nltk.corpus.nps_chat.xml_posts()[:10000]
posts = nltk.corpus.nps_chat.xml_posts()
output_path = '../data/sentence-type-data-nltk.csv'

def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

lst = []
for post in posts:
    lst.append([post.text, post.get('class')])
    # print(post.text, post.get('class'))

df = pd.DataFrame(lst, columns=['sentence', 'type'])
df.to_csv(output_path, index=False)

featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
# classifier = nltk.NaiveBayesClassifier.train(train_set)
# print(nltk.classify.accuracy(classifier, test_set))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


import seaborn as sns
import pandas as pd
import spacy
import re
import nltk
import string
import math
import numpy as np

# import matplotlib
# import matplotlib.pyplot as plt

# from YoutubeSearch import YoutubeSearch

nlp = spacy.load('en_core_web_sm')
# data_path = '../data/wine.csv'


class KMeansClusterer:

    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    # def get_data(self, data_path):
    #     df = pd.read_csv(data_path)
    #     df = df.iloc[:1000]
    #
    #     df = df.dropna(subset=['description'])
    #     return df

    def create_df(self, texts):
        df_src = {"description": texts}
        df = pd.DataFrame(df_src)
        return df

    def elbow_method(self, X, max_k):

        Sum_of_squared_distances = []
        K = range(1, max_k)
        for k in K:
            km = KMeans(init="k-means++", n_clusters=k)
            km = km.fit(X)
            Sum_of_squared_distance = [k, km.inertia_]
            Sum_of_squared_distances.append(Sum_of_squared_distance)
        return Sum_of_squared_distances

    def get_silhouette_score(self, X, k):
        silhouette_score_array = []
        for n_clusters in range(2, k):
            clusterer = KMeans(init="k-means++", n_clusters=n_clusters, random_state=42)
            y = clusterer.fit_predict(X)

            # message = "For n_clusters = {} The average silhouette_score is: {}"
            # print(message.format(n_clusters, silhouette_score(X, y)))
            silhouette_score_array.append([n_clusters, silhouette_score(X, y)])
        return silhouette_score_array

    def get_silhouette_optimal_cluster_size(self, data_array):
        max_score = -1.1 #score can lie between -1 to 1.
        max_score_cluster_size = 2
        for k, score in data_array:
            if (score>max_score):
                max_score = score
                max_score_cluster_size = k
        return max_score_cluster_size

    # def get_varietals(self, df):
    #     varietals = ' '.join(df.variety.unique().tolist()).lower()
    #     varietals = re.sub('-', ' ', varietals)
    #     varietals = varietals.split()
    #     return varietals

    def clean_string(self, text):

        final_string = ""
        # Make lower
        text = text.lower()
        # Remove line breaks
        text = re.sub('\\n', '', text)
        # Remove puncuation
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        # Lemmatize
        text = nlp(text)
        text = [y.lemma_ for y in text]
        # Convert to list
        # text = text.split()
        # Remove stop words
        useless_words = nltk.corpus.stopwords.words("english")
        # useless_words = useless_words + varietals + ['drink', 'wine']

        text_filtered = [word for word in text if not word in useless_words]
        # Remove numbers
        text_filtered = [re.sub('\\w*\\d\\w*', '', w) for w in text_filtered]
        final_string = ' '.join(text_filtered)

        return final_string

    def clean_data(self, df):
        # varietals = self.get_varietals(df)
        df['description'].replace('', np.nan, inplace=True)
        df = df.dropna()
        df['description_clean'] = df['description'].apply(lambda x: self.clean_string(x))
        return df

    def encode_text(self, df):
        X = self.vectorizer.fit_transform(df['description_clean'])
        return X

    def fit_model(self, cluster_size, data_array):
        model = KMeans(init="k-means++", n_clusters=cluster_size, max_iter=25, n_init=1)
        model.fit(data_array)
        return model

    def cluster_details(self, model, encoded_text, df, cluster_size):
        clust_labels = model.predict(encoded_text)
        # cent = model.cluster_centers_

        kmeans_labels = pd.DataFrame(clust_labels)
        df.insert((df.shape[1]), 'clusters', kmeans_labels)
        # And finally, let's build a quick data frame that shows the top 15 words from each of the two clusters and see what we get.

        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = self.vectorizer.get_feature_names_out()

        results_dict = {}

        for i in range(cluster_size):
            terms_list = []

            for ind in order_centroids[i, :15]:
                terms_list.append(terms[ind])

            results_dict[f'Cluster {i}'] = terms_list

        # df_clusters = pd.DataFrame.from_dict(results_dict)
        # print(df_clusters.head())
        return df

    def combine_cluster(self, df, cluster_col, text_col, cluster_size):
        res = []
        for index in range(cluster_size):
            x = list(df[df[cluster_col]==index][text_col])
            res.append(x)
        return res

    def clusterize_texts(self, texts):
        df = self.create_df(texts)
        df = self.clean_data(df)
        cluster_range = math.ceil(len(df) / 2) + 1
        # print(len(df), cluster_range)
        X = self.encode_text(df)
        silhouette_scores = self.get_silhouette_score(X, cluster_range)
        # print(silhouette_scores)
        cluster_size = self.get_silhouette_optimal_cluster_size(silhouette_scores)
        print("optimal_cluster_size - ", cluster_size)
        model = self.fit_model(cluster_size, X)
        df = self.cluster_details(model, X, df, cluster_size)
        # print(df)
        combined_text = self.combine_cluster(df, 'clusters', 'description', cluster_size)
        # print(combined_text)
        return combined_text


if __name__ == "__main__":

    # ys = YoutubeSearch()


    km = KMeansClusterer()
    texts = ['Happy durga puja sir', 'Happy Durga Puja Happy Panchami',
             'You may not have seen Chor Bagan...Near Metro...One of the Finest Pandal Bet',
             'Dada, you from North Kolkata, will visit another big Puja, now for Dada or Sangha Baranagar. This is the Introduction that I guarantee you will be impressed by the time you watch it.',
             'Coming to kolkata on 5th oct, kindly guide us which pandal to visit.. as its last day',
             'Those who keep Chorbagans top 10', 'Durga Puja video ', '', 'Kalyani, West Bengal, Nadia district',
             'Jai maa Durga ', 'Jai Maa Durga . Har Har Mahadev ', 'Bhai background music download from Katha',
             'Hope You Enjoyed The Video Add Me on Social Media Instagram',
             'Dhone dhonne puspe vora. Its poem on rabindranath thakur.',
             'Thanks for watching Add Me on Social Media Instagram', 'Jay maa durga ',
             'Dada I am Shubhjit Paul saying how to contact', 'Jai maa Durga Jai maa Durga ', 'Jay eyes on ']
    clustered_texts = km.clusterize_texts(texts)
    print(clustered_texts)
    '''
    cluster_range = 20
    # df = km.get_data(data_path=data_path)
    data = ['Happy durga puja sir', 'Happy Durga Pujo Shubh Panchami', 'You probably havent seen Chor Bagan...near. mg metro.. one of finest pandal I bet', 'Dada, wha north Kolkata, another and big Puja visit Will do it, Nav para dada Y Sangha Baranagar. Ehaka thim hei Introduction Look, I guarantee it. ki you like it', 'Coming to kolkata on 5th oct, kindly guide us which pandal to visit.. as its last day', 'Chorbagan ta top 10 a It would be better to keep it', 'Durga Puja video ', '', 'Kalyani, West Bengal, Nadia district', 'go mom Durga ', 'I say, grandpa, your drone is fine now', 'Kolkata is most important city in India ', 'Patna ka', 'Kharagpur Durga Puja Pandal 2024', 'Hope You Enjoyed The Video Add Me on Social Media Instagram', 'Dhono dhonne puspe vora. It,s poem on rabindra nath thakur.', 'Thanks for watching Add Me on Social Media Instagram', 'Jay maa durga ', 'Dada ami I am saying Shubojit Paul contact a basket ki lures', 'go mom Durga Jai maa Durga ', 'Jay eye di ']
    df = pd.DataFrame(data, columns=['description'])
    # varietals = km.get_varietals(df)
    vectorizer = TfidfVectorizer()
    df = km.clean_data(df)
    X = km.encode_text(vectorizer, df)
    silhouette_scores = km.get_silhouette_score(X, cluster_range)
    cluster_size = km.get_silhouette_optimal_cluster_size(silhouette_scores)

    model = km.fit_model(cluster_size, X)

    k = cluster_size
    df = km.cluster_details(X, df, k)

    combined_text = km.combine_cluster(df, 'clusters', 'description', cluster_size)
    print("Cluster size - ", cluster_size)
    for txt in combined_text:
        print("-----------------")
        print(txt)
        wrapped_text = ys.wrap_text(txt)
        summary = ys.text_summarizer.summarizeText(wrapped_text)
        summary = str(summary).replace(" .", ".")
        print(summary)
    # print(combined_text)
    new_docs = ['Rich deep color of oak and chocolate.',
                'Light and crisp with a hint of vanilla.',
                'Hints of citrus and melon.',
                'Dark raspberry and black cherry flavors.']

    pred = model.predict(vectorizer.transform(new_docs))
    '''

    '''
    # df = km.create_df(texts)
    # df = km.clean_data(df)
    # cluster_range = math.ceil(len(df)/2)+1
    # print(len(df), cluster_range)
    # X = km.encode_text(df)
    # silhouette_scores = km.get_silhouette_score(X, cluster_range)
    # print(silhouette_scores)
    # cluster_size = km.get_silhouette_optimal_cluster_size(silhouette_scores)
    # print(cluster_size)
    # model = km.fit_model(cluster_size, X)
    # df = km.cluster_details(model, X, df, cluster_size)
    # print(df)
    # combined_text = km.combine_cluster(df, 'clusters', 'description', cluster_size)
    # print(combined_text)
    # print(df)
    '''


    # print(pred)



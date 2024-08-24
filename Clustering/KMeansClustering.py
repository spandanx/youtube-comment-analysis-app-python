from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


import seaborn as sns
import pandas as pd
import spacy
import re
import nltk
import string
import numpy as np

import matplotlib
import matplotlib.pyplot as plt



nlp = spacy.load('en_core_web_sm')
data_path = '../data/wine.csv'


class KMeansClusterer:
    def get_data(self, data_path):
        df = pd.read_csv(data_path)
        df = df.iloc[:1000]

        df = df.dropna(subset=['description'])
        return df

    def elbow_method(self, X, max_k):

        Sum_of_squared_distances = []
        K = range(1, max_k)
        for k in K:
            km = KMeans(init="k-means++", n_clusters=k)
            km = km.fit(X)
            Sum_of_squared_distance = [k, km.inertia_]
            Sum_of_squared_distances.append(Sum_of_squared_distance)

        # ax = sns.lineplot(x=K, y=Sum_of_squared_distances)
        # ax.lines[0].set_linestyle("--")

        # Add a vertical line to show the optimum number of clusters
        # plt.axvline(2, color='#F26457', linestyle=':')
        #
        # plt.xlabel('k')
        # plt.ylabel('Sum of Squared Distances')
        # plt.title('Elbow Method For Optimal k')
        # plt.show()
        return Sum_of_squared_distances

    def get_silhouette_score(self, X, k):
        silhouette_score_array = []
        for n_clusters in range(2, k):
            clusterer = KMeans(init="k-means++", n_clusters=n_clusters, random_state=42)
            y = clusterer.fit_predict(X)

            message = "For n_clusters = {} The average silhouette_score is: {}"
            print(message.format(n_clusters, silhouette_score(X, y)))
            silhouette_score_array.append([n_clusters, silhouette_score(X, y)])
        return silhouette_score_array

    def get_silhouette_cluster_size(self, data_array):
        max_score = -1.1 #score can lie between -1 to 1.
        max_score_cluster_size = 2
        for k, score in data_array:
            if (score>max_score):
                max_score = score
                max_score_cluster_size = k
        return max_score_cluster_size

    def get_varietals(self, df):
        varietals = ' '.join(df.variety.unique().tolist()).lower()
        varietals = re.sub('-', ' ', varietals)
        varietals = varietals.split()
        return varietals

    def clean_string(self, text, varietals):

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
        useless_words = useless_words + varietals + ['drink', 'wine']

        text_filtered = [word for word in text if not word in useless_words]
        # Remove numbers
        text_filtered = [re.sub('\\w*\\d\\w*', '', w) for w in text_filtered]
        final_string = ' '.join(text_filtered)

        return final_string

    def clean_data(self, df):
        varietals = self.get_varietals(df)
        df['description_clean'] = df['description'].apply(lambda x: self.clean_string(x, varietals))
        return df

    def encode_text(self, vectorizer, df):
        X = vectorizer.fit_transform(df['description_clean'])
        return X

    def fit_model(self, cluster_size, data_array):
        model = KMeans(init="k-means++", n_clusters=cluster_size, max_iter=25, n_init=1)
        model.fit(data_array)
        return model

    def cluster_details(self, X, df, k):
        clust_labels = model.predict(X)
        cent = model.cluster_centers_

        kmeans_labels = pd.DataFrame(clust_labels)
        df.insert((df.shape[1]), 'clusters', kmeans_labels)
        # And finally, let's build a quick data frame that shows the top 15 words from each of the two clusters and see what we get.

        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()

        results_dict = {}

        for i in range(k):
            terms_list = []

            for ind in order_centroids[i, :15]:
                terms_list.append(terms[ind])

            results_dict[f'Cluster {i}'] = terms_list

        df_clusters = pd.DataFrame.from_dict(results_dict)
        print(df_clusters.head())
        return df


st = ['Rich deep color of oak and chocolate.',
            'Light and crisp with a hint of vanilla.',
            'Hints of citrus and melon.',
            'Dark raspberry and black cherry flavors.']



if __name__ == "__main__":
    cluster_range = 10
    km = KMeansClusterer()
    df = km.get_data(data_path=data_path)
    # varietals = km.get_varietals(df)
    vectorizer = TfidfVectorizer()
    df = km.clean_data(df)
    X = km.encode_text(vectorizer, df)
    silhouette_scores = km.get_silhouette_score(X, cluster_range)
    cluster_size = km.get_silhouette_cluster_size(silhouette_scores)

    model = km.fit_model(cluster_size, X)

    k = cluster_size
    df = km.cluster_details(X, df, k)

    new_docs = ['Rich deep color of oak and chocolate.',
                'Light and crisp with a hint of vanilla.',
                'Hints of citrus and melon.',
                'Dark raspberry and black cherry flavors.']

    pred = model.predict(vectorizer.transform(new_docs))
    print(pred)



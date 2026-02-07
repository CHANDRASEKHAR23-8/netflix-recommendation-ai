import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("netflix_titles.csv")
df.fillna("", inplace=True)

df["bag"] = df["title"] + " " + df["director"] + " " + df["cast"] + " " + df["listed_in"] + " " + df["description"]

tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(df["bag"])

svd = TruncatedSVD(n_components=100)
X_reduced = svd.fit_transform(X)

kmeans = KMeans(n_clusters=10, random_state=42)
df["cluster"] = kmeans.fit_predict(X_reduced)

plt.figure(figsize=(8,6))
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=df["cluster"])
plt.title("Netflix Content Clusters")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

similarity = cosine_similarity(X)

def recommend(title):
    idx = df[df["title"] == title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    for i in scores:
        print(df.iloc[i[0]]["title"])

movie = input("Enter a movie name: ")
recommend(movie)

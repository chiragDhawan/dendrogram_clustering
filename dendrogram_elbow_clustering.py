from pyspark import SparkContext, HiveContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf, StringType
import re
import subprocess
import sys
import traceback as tb
from sklearn.externals import joblib
import nltk
import re
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sc = SparkContext()

hc = HiveContext(sc)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

sc.setLogLevel("ERROR") ###setting so that there is no verbose

def getData():
    rawDataDf = hc.sql(" ") ###Enter the select query to get data from hive
    data=rawDataDf.toPandas()

    return data

def tokenize_only(document):
    return(document.split(' '))

def tfIdf_transform(document):
    tfidf_vect = TfidfVectorizer(max_df=0.9, use_idf=False) ##tfidf with max_df 0.9 which tells the max limit after which it would not take the term
    tfidf_matrix = tfidf_vect.fit_transform(document)
    print(tfidf_matrix.shape)

    return tfidf_matrix

def cleanup_data(document):
    documents_with_only_letters = re.sub("[^a-zA-Z]", " ", document)  # remove eveything except for letters
    documents_with_only_letters = documents_with_only_letters.lower()
    return documents_with_only_letters

def count_transform(document):
    print(document.head())
    count_vect=CountVectorizer()
    count_matrix=count_vect.fit_transform(document)

    return count_matrix

def cleanup_data(document):
    documents = document.lower()  # converting everything to lowercase
    words = documents.split(' ')
    words = " ".join(words)  # join the words to make them a sentence once again
    return words

def distance_matrix(matrix):
    #if similarity is required then using cosine similarity
    sim=cosine_similarity(matrix)
    #if distance is required then using 1-cosine
    distance = 1 - cosine_similarity(matrix)
    return sim

### Using K-means first ### It will use sum of squares to as the distance matrix
def k_means_fit(matrix):
    ##giving random no. of clusters
    num_clusters = 4
    #matrix.reshape(-1,1)
    km=KMeans(n_clusters=num_clusters)
    return km.fit_predict(matrix)

def k_means_plot(matrix):
    Ks = range(1, 10)
    km = [KMeans(n_clusters=i) for i in Ks]
    score = [km[i].fit(matrix).score(matrix) for i in range(len(km))]
    plt.title("Kmeans score")
    plt.plot(Ks,score)
    plt.savefig("Kmeans_from.png")
    ### plot gives elbow at 4 cluster

def get_Model():
    km=joblib.load("cluster_docs_full.pkl")
    return km

def get_prediction(km,data):
    clusters=km.labels_
    assignments=pd.Series(clusters,index=data.index)



from scipy.cluster.hierarchy import linkage, dendrogram

def heirarchical_cluster_fit(matrix):
    linkage_matrix=linkage(matrix.todense(),method='ward') ##heirarchical clustering using the linkage method to create the dendrogram
    return linkage_matrix

from scipy.cluster.hierarchy import fcluster as fcl
def create_first_table():
    linkage_mtx=joblib.load('linkage_matrix_heirarchical_with_cluster.pkl')
    lb=fcl(linkage_mtx,8,'maxclust')
    data=hc.sql("")
    data=data.toPandas()
    data['cluster_label']=lb
    df=hc.createDataFrame(data)
    df.write.mode("append").saveAsTable("")

def create_another_table(matrix):
    labels=k_means_fit(matrix)
    data = hc.sql("")
    data = data.toPandas()
    data['cluster_label']=labels
    df = hc.createDataFrame(data)
    df.write.mode("append").saveAsTable("")

create_another_table(tfIdf_transform(getData()))
#############################################################################################

def group_data_and_transform_cluster(data):
    grpby = data.groupby('cluster_labels')
    df_dict = dict()
    cluster_predict_matrix=[]
    for grps in grpby.groups:
        df_dict[grps] = grpby.get_group(grps)
        cluster_predict_matrix.append(heirarchical_cluster_fit(tfIdf_transform(grpby.get_group(grps)["parsed_where"].unique())))
    for i in range(0, len(cluster_predict_matrix)):
        joblib.dump(cluster_predict_matrix[i],'where_cluster_%s.pkl'%i)


def get_dendrogram_from_link_matrix(link_matrix):
    for i in range(0,len(link_matrix)):
        plt.title('where_cluster_%s'%i)
        dendrogram(
            link_matrix[i],
            truncate_mode='lastp',  # show only the last p merged clusters
            p=40,  # show only the last p merged clusters
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,  # to get a distribution impression in truncated branches
        )
dendrogram(link_matrix[i])

plt.savefig('where_cluster_%s.png'%i, dpi=200)

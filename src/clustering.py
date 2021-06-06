from sklearn.cluster import KMeans, AgglomerativeClustering
from src.misc import *
import numpy as np
import hdbscan
from typing import Tuple
import time


def k_means_clustering(word_embeddings: list, word_weights: list = None, params: dict = None) -> list:
    """
    k_means_clustering performs K-Means clustering on word_embeddings

    :param word_embeddings: list of word embeddings
    :param word_weights: list of word weights
    :param params: dictionary of clustering weights

    :return: list of labels
    """
    model = KMeans(**params)
    return model.fit_predict(word_embeddings, word_weights)


def agglomerative_clustering(word_embeddings: list, word_weights: list = None, params: dict = None) -> list:
    """
    agglomerative_clustering performs Agglomerative clustering on the given word_embeddings

    :param word_embeddings: list of word embeddings
    :param word_weights: list of word weights, for weighted clustering
    :param params: cluster parameters

    :return: list of labels
    """
    model = AgglomerativeClustering(**params)
    return model.fit_predict(word_embeddings, word_weights)


def hdbscan_clustering(words: list, embeddings: list, min_cluster_size: int = 10,  n_words: int = 30) \
        -> Tuple[list, list, float]:
    """
    hdbscan_clustering performs HDBSCAN clustering on the list of embeddings

    :param words: list of words
    :param embeddings: list of embeddings of the words
    :param min_cluster_size: minimum cluster size
    :param n_words: number of topic representatives

    :return:
        - hdbscan_clusters_words - list of topic representatives
        - hdbscan_clusters_words_embeddings - list of embeddings of topic representatives
        - execution_time_hdbscan - process time needed for execution
    """
    assert len(words) == len(embeddings)
    start_time_hdbscan = time.process_time()

    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean',
                              cluster_selection_method='eom').fit(embeddings)

    temp_cluster_words = [[] for _ in range(len(set(cluster.labels_)) - 1)]
    temp_cluster_embeddings = [[] for _ in range(len(set(cluster.labels_)) - 1)]
    for i, label in enumerate(cluster.labels_):

        if label == -1:
            # noise
            continue
        temp_cluster_words[label].append([words[i], cluster.probabilities_[i]])
        temp_cluster_embeddings[label].append(embeddings[i])

    hdbscan_clusters_words = []
    hdbscan_clusters_words_embeddings = []
    for i_c, c in enumerate(temp_cluster_words):
        c_sorted_indices = sorted(range(len(c)), key=lambda i_w: c[i_w][1], reverse=True)

        hdbscan_clusters_words.append([c[i][0] for i in c_sorted_indices[:n_words]])
        hdbscan_clusters_words_embeddings.append([temp_cluster_embeddings[i_c][i]
                                                  for i in c_sorted_indices[:n_words]])

    execution_time_hdbscan = time.process_time() - start_time_hdbscan
    return hdbscan_clusters_words, hdbscan_clusters_words_embeddings, execution_time_hdbscan


def sort_words(processed_segments: list, cluster_words: list, cluster_embeddings: list,
               weight_type: str = "tf") -> Tuple[list, list]:
    """
    sort_words sorts the words within each cluster by the given weight type

    :param processed_segments: list of preprocessed segments
    :param cluster_words: list of word clusters
    :param cluster_embeddings: list of word embedding clusters
    :param weight_type: weighted type by which the words are sorted ["tf", "tf-df", "tf-idf"]

    :return:
        - sorted_cluster_words - sorted topic representatives
        - sorted_cluster_embeddings - sorted topic embeddings

    """

    assert len(cluster_words) == len(cluster_embeddings), "cluster_words and cluster_embeddings do not " \
                                                          "have the same amount of clusters"
    assert all([len(cluster_words[i]) == len(cluster_embeddings[i]) for i in range(len(cluster_words))]), (
        "each cluster must have the same number of words and embeddings")

    assert weight_type in ["tf", "tf-df", "tf-idf"], "wrong counter_type!"

    # word_weight = None
    n_words = len([w for doc in processed_segments for w in doc])
    clusters_vocab = list(set([w for c_words in cluster_words for w in c_words]))

    word_weights = get_word_weights(processed_segments, vocab=clusters_vocab, n_words=n_words, weight_type=weight_type)

    # calculate cluster centers
    cluster_centers = [np.mean(c_embeddings, axis=0) for c_embeddings in cluster_embeddings]
    assert len(cluster_centers) == len(cluster_embeddings)
    assert len(cluster_centers[0]) == len(cluster_embeddings[0][0])

    # sort cluster_words
    sorted_cluster_words = []
    sorted_cluster_embeddings = []
    sorted_cluster_idxs = [sorted(range(len(c)),
                                  key=lambda k: word_weights[c[k]],
                                  reverse=True)
                           for i_c, c in enumerate(cluster_words)]

    for i_c in range(len(cluster_words)):
        sorted_cluster_words.append([cluster_words[i_c][i] for i in sorted_cluster_idxs[i_c]])
        sorted_cluster_embeddings.append([cluster_embeddings[i_c][i] for i in sorted_cluster_idxs[i_c]])

    return sorted_cluster_words, sorted_cluster_embeddings


def get_word_clusters(processed_docs: list, words: list, word_embeddings: list, vocab: list,
                      clustering_type: str, params: dict, clustering_weight_type: str = 'tf',
                      ranking_weight_type=None) -> Tuple[list, list, float]:
    """
    get_word_clusters returns a sorted list of words for each cluster

    :param processed_docs: list of preprocessed documents
    :param words: list of words
    :param word_embeddings: list of word embeddings
    :param vocab: list of vocabulary words
    :param clustering_type: defines the clustering method ('K-Means')
    :param params: clustering parameters
    :param clustering_weight_type: word weighting type used for clustering ("tf", "tf-df", "tf-idf")
    :param ranking_weight_type: word weighting type used for ranking words ("tf", "tf-df", "tf-idf")

    :return:
        - cleaned_cluster_words - list of cluster words for each cluster
        - cleaned_cluster_embeddings - list of word embeddings for each cluster (sorted!)
        - execution_time - process time needed for execution
    """
    # :param n_words: number of words for every cluster

    assert len(word_embeddings) == len(words), "word_embeddings and word list do not have the same length"
    assert clustering_type in ['K-Means', 'Agglomerative'], "incorrect clustering_type"
    assert all([w in vocab for w in words]), "some words are not in the vocabulary"

    start_time = time.process_time()

    clustering_dict = {
        'K-Means': k_means_clustering,
        'Agglomerative': agglomerative_clustering
    }

    if clustering_weight_type is None:
        print("Performing clustering without any weights!")

        word_weights = None

    else:

        n_words = len([w for doc in processed_docs for w in doc])
        word_weights_dict = get_word_weights(processed_docs, vocab, n_words, weight_type=clustering_weight_type)
        word_weights = [word_weights_dict[w] for w in words]

    # cluster words to cluster labels
    labels = clustering_dict[clustering_type](word_embeddings, word_weights, params)

    # assign each word to cluster list
    cluster_words = [[] for _ in range(len(set(labels)))]
    cluster_embeddings = [[] for _ in range(len(cluster_words))]
    for l_id, label in enumerate(list(labels)):

        w = words[l_id]
        if w not in vocab:
            continue

        cluster_words[label].append(w)
        cluster_embeddings[label].append(word_embeddings[l_id])

    # remove clusters with <= 5 words:
    cleaned_cluster_words = []
    cleaned_cluster_embeddings = []
    for i_c, c in enumerate(cluster_words):

        if len(c) <= 5:
            continue
        cleaned_cluster_words.append(c)
        cleaned_cluster_embeddings.append(cluster_embeddings[i_c])

    # if no clusters have >= 6 words
    if len(cleaned_cluster_words) == 0:
        cleaned_cluster_words.append([w for c in cluster_words for w in c])
        cleaned_cluster_embeddings.append([emb for c in cluster_embeddings for emb in c])

    execution_time = time.process_time() - start_time

    if ranking_weight_type is None:
        return cleaned_cluster_words, cleaned_cluster_embeddings, execution_time

    else:
        # using re-ranking model instead of topic vector similarity
        sorted_cluster_words, sorted_cluster_embeddings = sort_words(processed_docs, cleaned_cluster_words,
                                                                     cleaned_cluster_embeddings,
                                                                     weight_type=ranking_weight_type)
        return sorted_cluster_words, sorted_cluster_embeddings, execution_time

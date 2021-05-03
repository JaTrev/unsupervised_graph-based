from src.vectorization import *
from src.clustering import *
from src.graphs import *
from pathlib import Path
import pickle


def w_d_clustering(data_processed: list, data_set_name: str, vocab: list,
                   segment_labels_true: list, test_tokenized_segments: list, segment_embedding_type="w2v_avg",
                   x: list = None, true_topic_amount=10):
    """
    w_d_clustering using document embedding and word embeddings to extract the corpus topics

    :param data_processed: list of preprocessed segments
    :param data_set_name: preprocessed data set name
    :param vocab: vocabulary word list
    :param segment_labels_true: list of segment labels
    :param test_tokenized_segments:
    :param segment_embedding_type:
    :param x: list of number of topics values
    :param true_topic_amount: true number of topics (must be included in x)

    """
    # main extrinsic evaluation metric: ARI
    # https://stats.stackexchange.com/questions/381223/evaluation-of-clustering-method

    assert data_set_name in ["JN", "FP"]

    # create x-values (number of topics list)
    if x is None:
        x = list(range(2, 22, 2))
        assert true_topic_amount in x
    else:
        assert isinstance(x, list)

    # Word2Vec and clustering parameters
    if segment_embedding_type in ["w2v_avg", "w2v_sum"]:
        params = {"min_c": 10, "win": 7, "negative": 0, "sample": 1e-5, "hs": 1, "epochs": 400, "sg": 1, 'seed': 42,
                  'ns_exponent': 0.75}

        if data_set_name == "FP":
            # Fully Preprocessed: 9
            min_cluster_size = 9

        else:
            # Just Nouns: min cluster size =  7
            min_cluster_size = 7

    else:
        # Doc2Vec and clustering parameters
        assert segment_embedding_type == "doc2vec"
        params = {"min_c": 10, "win": 5, "negative": 30, "sample": 1e-5, "hs": 0, "epochs": 400, 'seed': 42,
                  'ns_exponent': 0.75, "dm": 0, "dbow_words": 1}
        min_cluster_size = 8

    dict_file = "semantic-space-dict-" + segment_embedding_type + "-" + data_set_name + ".pickle"
    if Path("data/" + dict_file).is_file():

        print("using pre-calculated file")
        with open("data/" + dict_file, "rb") as myFile:
            temp_dict = pickle.load(myFile)

            doc_data = temp_dict['doc_data']
            short_true_labels = temp_dict['short_true_labels']
            doc_embeddings = temp_dict['doc_embeddings']
            vocab_words = temp_dict['vocab_words']
            vocab_embeddings = temp_dict['vocab_embeddings']

    else:

        doc_data, short_true_labels, doc_embeddings, vocab_words, vocab_embeddings = semantic_space_embeddings(
            data_processed, segment_labels_true, vocab, segment_embedding_type, params)

        temp_dict ={'doc_data': doc_data, 'short_true_labels': short_true_labels, 'doc_embeddings': doc_embeddings,
                    'vocab_words': vocab_words, 'vocab_embeddings': vocab_embeddings}
        with open("data/" + dict_file, "wb") as myFile:
            pickle.dump(temp_dict, myFile)

    y_topics = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}
    y_c_v_model = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}
    y_npmi_model = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}
    test_y_c_v_model = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}
    test_y_npmi_model = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}
    doc_topics_pred_model = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}
    doc_topics_true_model = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}

    # perform HDBSCAN clustering
    hdbscan_clustering_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean',
                                               cluster_selection_method='eom').fit(doc_embeddings)
    labels = hdbscan_clustering_model.labels_

    hdbscan_clusters_docs_embeddings = [[] for _ in range(len(set(labels)) - 1)]
    hdbscan_labels_predict = []
    hdbscan_true_labels = []
    for i, label in enumerate(labels):

        if label == -1:
            # noise
            continue
        hdbscan_clusters_docs_embeddings[label].append(doc_embeddings[i])
        hdbscan_true_labels.append(short_true_labels[i])
        hdbscan_labels_predict.append(label)

    # iterate over x-values (number of topics list)
    for k in x:

        # iterate over clustering methods
        for cluster_type in y_topics.keys():

            if cluster_type == "HDBSCAN":
                clusters_docs_embeddings = hdbscan_clusters_docs_embeddings
                labels_predict = hdbscan_labels_predict
                true_labels = hdbscan_true_labels

            else:
                if cluster_type == "K-Means":
                    clustering_params = {'n_clusters': k, 'random_state': 42, }
                else:
                    clustering_params = {'n_clusters': k}

                # clustering document via document embeddings
                _, clusters_docs_embeddings, labels_predict, _ = segment_clustering(
                    doc_data, doc_embeddings, cluster_type, params=clustering_params)

                true_labels = short_true_labels

            # calulate topic vectors and use cosine similarity to find topic representatives
            topic_embeddings = []
            topics_words = []
            topics_words_embeddings = []
            for docs_embeddings in clusters_docs_embeddings:

                t_embedding = np.average(docs_embeddings, axis=0)
                t_embedding_sim_matrix = cosine_similarity(t_embedding.reshape(1, -1), vocab_embeddings)[0]
                most_sim_ids = np.argsort(t_embedding_sim_matrix, axis=None)[:: -1]

                t_words = [vocab_words[i] for i in most_sim_ids[:10]]
                t_words_embeddings = [vocab_embeddings[i] for i in most_sim_ids[:10]]

                topics_words.append(t_words)
                topics_words_embeddings.append(t_words_embeddings)
                topic_embeddings.append(t_embedding)

            # save topics
            y_topics[cluster_type].append(topics_words)

            # topic model evaluation
            # intrinsic scores
            y_c_v_model[cluster_type].append(c_v_coherence_score(data_processed, topics_words, cs_type='c_v'))
            y_npmi_model[cluster_type].append(npmi_coherence_score(data_processed, topics_words, len(topics_words)))

            # extrinsic scores
            if test_tokenized_segments is not None:
                test_y_c_v_model[cluster_type].append(c_v_coherence_score(test_tokenized_segments, topics_words,
                                                                          cs_type='c_v'))
                test_y_npmi_model[cluster_type].append(npmi_coherence_score(test_tokenized_segments, topics_words,
                                                                            len(topics_words)))
            else:
                test_y_c_v_model[cluster_type].append(-1000.0)
                test_y_npmi_model[cluster_type].append(-1000.0)

            if k == true_topic_amount:
                label_distribution(true_labels, labels_predict, cluster_type)

            # save predicted topics assigned for classification evaluation
            doc_topics_pred_model[cluster_type].append(labels_predict)
            doc_topics_true_model[cluster_type].append(true_labels)

    # save clustering scores
    save_model_scores(x_values=x, models=list(y_topics.keys()), model_topics=y_topics, model_c_v_scores=y_c_v_model,
                      model_npmi_scores=y_npmi_model, model_c_v_test_scores=test_y_c_v_model,
                      model_npmi_test_scores=test_y_npmi_model, filename_prefix='JESS')

    # save classification performance of every model
    for m in list(y_topics.keys()):
        vis_classification_score(y_topics[m], m, doc_topics_true_model[m], doc_topics_pred_model[m],
                                 filename="visuals/classification_scores_" + str(m) + ".txt",
                                 multiple_true_label_set=True)

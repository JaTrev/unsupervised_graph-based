from src.vectorization import *
from src.clustering import *
from src.graphs import *
from src.misc import save_model_scores
import pickle
from networkx.algorithms import approximation as apxa
from pathlib import Path
import time

w2v_params = {"min_c": 10, "win": 7, "negative": 0, "sample": 1e-5, "hs": 1, "epochs": 400, "sg": 1, 'seed': 42,
              'ns_exponent': 0.75}


def word2vec_topic_model(data_processed: list, vocab: list, tokenized_docs: list, test_tokenized_segments: list,
                         data_set_name: str, x: list = None, topic_vector_flag: bool = False):
    """
    word2vec_topic_model performs topic modeling in word space using Word2Vec embeddings,
    performing either TVS model or RRW model.
    The function produces a range of files that list the resulting topics and visualize the model's performance.

    :param data_processed: preprocessed data set used to calculated word embeddings
    :param vocab: vocabulary of the preprocessed data set
    :param tokenized_docs: tokenized version of the training data set
    :param test_tokenized_segments: tokenized version of the test data set
    :param data_set_name: preprocessed data set used
    :param x: list of number of topics to iterate over, default: list(range(2, 22, 2))
    :param topic_vector_flag: flag used to switch between TVS model and RRW and, default: False (RRW model)

    """
    assert data_set_name in ["JN", "FP"]

    # set weighting strategies
    clustering_weight_type = 'tf'
    ranking_weight_type = 'tf'

    # create x-values (number of topics list)
    if x is None:
        x = list(range(2, 22, 2))

    else:
        assert isinstance(x, list)

    y_c_v_model = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}
    y_dbs_model = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}
    y_npmi_model = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}
    test_y_c_v_model = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}
    test_y_npmi_model = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}
    y_topics = {'K-Means': [], 'Agglomerative': [], 'HDBSCAN': []}

    y_uMass_model = {'K-Means': [], 'Agglomerative': [], 'HDBSCAN': []}
    test_y_uMass_model = {'K-Means': [], 'Agglomerative': [], 'HDBSCAN': []}

    # get words and word embeddings
    # words, word_embeddings, _ = get_word_vectors(data_processed, vocab, params=w2v_params)

    words, word_embeddings = get_vocabulary_embeddings(data_processed, vocab, topic_model="baseline",
                                                       model_file_name="w2v_model-" + data_set_name + ".pickle")
    """
    w2v_model_file = "w2v_model-" + data_set_name + ".pickle"
    if Path("data/" + w2v_model_file).is_file():

        print("using pre-calculated w2v model")
        with open("data/" + w2v_model_file, "rb") as myFile:
            w2v_model = pickle.load(myFile)

            words = [w for w in vocab if w in w2v_model.wv.index2word]
            word_embeddings = [w2v_model.wv.vectors[w2v_model.wv.index2word.index(w)] for w in words]

    else:
        words, word_embeddings, w2v_model = get_word_vectors(data_processed, vocab, params=w2v_params)

        with open("data/" + w2v_model_file, "wb") as myFile:
            pickle.dump(w2v_model, myFile)
    """
    # perform HDBSCAN clustering
    start_time_hdbscan = time.process_time()
    _, _, hdbscan_clusters_words, hdbscan_clusters_words_embeddings = hdbscan_clustering(words, word_embeddings,
                                                                                         min_cluster_size=6)
    execution_time_hdbscan = time.process_time() - start_time_hdbscan

    # iterate over x-values (number of topics list)
    for k in x:

        # iterate over all topics
        for cluster_type in y_topics.keys():
            
            if cluster_type == "HDBSCAN":
                clusters_words = hdbscan_clusters_words
                clusters_words_embeddings = hdbscan_clusters_words_embeddings

                execution_time = execution_time_hdbscan
            else:
                assert cluster_type in ["K-Means", "Agglomerative"]
                if cluster_type == "K-Means":
                    clustering_params = {'n_clusters': k, 'random_state': 42, }
                else:
                    clustering_params = {'n_clusters': k}

                # re-ranking cluster words if RRW model
                if topic_vector_flag:
                    ranking_weight = None
                else:
                    ranking_weight = ranking_weight_type

                print("------------------------")
                print("clustering type: " + str(cluster_type))
                print("k: " + str(k))
                start_time = time.process_time()

                clusters_words, clusters_words_embeddings = get_word_clusters(
                    data_processed, words, word_embeddings, vocab, clustering_type=cluster_type,
                    params=clustering_params, clustering_weight_type=clustering_weight_type,
                    ranking_weight_type=ranking_weight,
                )

                execution_time = time.process_time() - start_time
                print("execution time: " + "{:.6f}".format(execution_time))
                print("------------------------")

            # perform Topic Vector Similarity
            if topic_vector_flag:
                topic_vectors = [get_topic_vector(c) for c in clusters_words_embeddings]

                # get topics based on topic vectors
                topic_vector_cluster_words = []
                topic_vector_cluster_words_embeddings = []
                for t_vector in topic_vectors:
                    sim_indices = get_nearest_indices(t_vector, word_embeddings)

                    topic_vector_cluster_words.append([words[i_w] for i_w in sim_indices])
                    topic_vector_cluster_words_embeddings.append([word_embeddings[i_w] for i_w in sim_indices])
                clusters_words = topic_vector_cluster_words

            y_topics[cluster_type].append(clusters_words)

            # topic model evaluation
            # intrinsic  scores
            y_c_v_model[cluster_type].append(get_coherence_score(tokenized_docs, clusters_words))
            y_npmi_model[cluster_type].append(npmi_coherence_score(tokenized_docs, clusters_words, len(clusters_words)))
            y_dbs_model[cluster_type].append(davies_bouldin_index(clusters_words_embeddings))
            y_uMass_model[cluster_type].append(get_coherence_score(tokenized_docs, clusters_words, cs_type='u_mass'))

            # extrinsic scores
            if test_tokenized_segments is not None:
                test_y_c_v_model[cluster_type].append(get_coherence_score(test_tokenized_segments, clusters_words))
                test_y_npmi_model[cluster_type].append(npmi_coherence_score(test_tokenized_segments, clusters_words,
                                                                            len(clusters_words)))
                test_y_uMass_model[cluster_type].append(
                    get_coherence_score(test_tokenized_segments, clusters_words, cs_type='u_mass'))
            else:
                test_y_c_v_model[cluster_type].append(-1000.0)
                test_y_npmi_model[cluster_type].append(-1000.0)
                test_y_uMass_model[cluster_type].append(-1000.0)

    # save model scores
    if topic_vector_flag:
        filename_prefix = "TVS"
    else:
        filename_prefix = "RRW"

    # save model scores

    save_model_scores(x_values=x, models=list(y_topics.keys()), model_topics=y_topics, model_c_v_scores=y_c_v_model,
                      model_npmi_scores=y_npmi_model, model_c_v_test_scores=test_y_c_v_model,
                      model_npmi_test_scores=test_y_npmi_model,
                      model_u_mass_scores=y_uMass_model, model_u_mass_test_scores=test_y_uMass_model,
                      execution_time=[-1 for _ in range(len(y_topics[0]))],
                      number_of_nodes=[-1 for _ in range(len(y_topics[0]))],
                      filename_prefix=filename_prefix,
                      model_dbs_scores=y_dbs_model, x_label="Percentile Cutoff")


def k_components_model(data_processed: list, vocab: list, tokenized_docs: list, test_tokenized_segments: list,
                       data_set_name: str, topic_vector_flag: bool = False):
    """
    k_components_model is used to perform topic model on the word embedding graph using k-components algorithm.
    This function uses the k-components approximation function from the Networkx library

    :param data_processed: preprocessed data set used to calculated word embeddings
    :param vocab: vocabulary of the preprocessed data set
    :param tokenized_docs: tokenized version of the training data set
    :param test_tokenized_segments: tokenized version of the test data set
    :param data_set_name: name of the preprocessed data set used
    :param topic_vector_flag: flag for using re-ranking words or topic vector similarity

    """
    n_words = len([w for d in data_processed for w in d])
    word_weights = get_word_weights(data_processed, vocab, n_words, weight_type='tf')

    vocab_words, vocab_embeddings, w2v_model = get_vocabulary_embeddings(data_processed, vocab,
                                                                         topic_model="k-components",
                                                                         model_file_name="w2v_model-k_components-"
                                                                                         + data_set_name + ".pickle")

    # dictionary used to save topic model scores
    y_topics = {"K=1": [], "K=2": [], "K=3": []}
    y_c_v_model = {"K=1": [], "K=2": [], "K=3": []}
    y_dbs_model = {"K=1": [], "K=2": [], "K=3": []}
    y_npmi_model = {"K=1": [], "K=2": [], "K=3": []}
    test_y_c_v_model = {"K=1": [], "K=2": [], "K=3": []}
    test_y_npmi_model = {"K=1": [], "K=2": [], "K=3": []}

    y_umass_model = {"K=1": [], "K=2": [], "K=3": []}
    test_y_umass_model = {"K=1": [], "K=2": [], "K=3": []}

    # iterate over all x values (percentile cutoff values)
    x = [x for x in range(50, 100, 10)] + [95]
    # x = [80]
    # x = [x for x in range(1, 11, 1)]
    execution_times = {"K=1": [], "K=2": [], "K=3": []}
    number_of_nodes = []

    for sim in x:

        # create word embedding graph using cutoff threshold
        graph, graph_creation_time = create_networkx_graph(vocab_words, vocab_embeddings,
                                                           similarity_threshold=0.8, percentile_cutoff=sim)

        number_of_nodes.append(graph.number_of_nodes())
        # calculate the k-components
        start_time = time.process_time()
        components_all = apxa.k_components(graph)
        k_components_time = time.process_time() - start_time

        # iterate over all k-components
        for k_component in y_topics.keys():

            # extract k-components
            temp_k_dict = {"K=1": 1, "K=2": 2, "K=3": 3}
            components = components_all[temp_k_dict[k_component]]

            # remove too small topics
            corpus_clusters = []
            clusters_words_embeddings = []
            for comp in components:
                if len(comp) >= 6:
                    corpus_clusters.append(list(comp))
                    clusters_words_embeddings.append([w2v_model.wv.get_vector(w) for w in comp])

            if topic_vector_flag:
                # perform Topic Vector Similarity
                topic_vectors = [get_topic_vector(c) for c in clusters_words_embeddings]

                # get topics based on topic vectors
                topic_vector_cluster_words = []
                topic_vector_cluster_words_embeddings = []
                for i, t_vector in enumerate(topic_vectors):
                    sim_indices = get_nearest_indices(t_vector, clusters_words_embeddings[i])

                    topic_vector_cluster_words.append([corpus_clusters[i][i_w] for i_w in sim_indices])
                    topic_vector_cluster_words_embeddings.append([clusters_words_embeddings[i][i_w]
                                                                  for i_w in sim_indices])
                cluster_words = topic_vector_cluster_words

            else:
                # sort topic representatives by node degree
                cluster_words = [sorted(list(c), key=(lambda w: sort_words_by(graph, w, word_weights)), reverse=True)
                                 for c in corpus_clusters]

            if len(cluster_words) <= 2:
                # topic model did not find enough topics
                # -1000.0 is the NaN value used in the charts, these values will not be shown in the charts
                cs_c_v = -1000.0
                dbs = -1000.0
                cs_npmi = -1000.0
                cs_c_v_test = -1000.0
                cs_npmi_test = -1000.0
                cs_u_mass = -1000.0
                cs_u_mass_test = -1000.0
            else:
                cluster_embeddings = [[w2v_model.wv.vectors[w2v_model.wv.key_to_index[w]] for w in words]
                                      for words in cluster_words]

                # topic model evaluation
                # intrinsic scores
                cs_c_v = get_coherence_score(tokenized_docs, cluster_words)
                dbs = davies_bouldin_index(cluster_embeddings)
                cs_npmi = npmi_coherence_score(data_processed, cluster_words, len(cluster_words))
                cs_u_mass = get_coherence_score(tokenized_docs, cluster_words, cs_type='u_mass')

                # extrinsic scores
                if test_tokenized_segments is not None:
                    cs_c_v_test = get_coherence_score(test_tokenized_segments, cluster_words)
                    cs_npmi_test = npmi_coherence_score(test_tokenized_segments, cluster_words, len(cluster_words))
                    cs_u_mass_test = get_coherence_score(test_tokenized_segments, cluster_words, cs_type='u_mass')
                else:
                    cs_c_v_test = -1000.0
                    cs_npmi_test = -1000.0
                    cs_u_mass_test = -1000.0

            y_topics[k_component].append(cluster_words)
            y_c_v_model[k_component].append(cs_c_v)
            y_npmi_model[k_component].append(cs_npmi)
            y_dbs_model[k_component].append(dbs)
            test_y_c_v_model[k_component].append(cs_c_v_test)
            test_y_npmi_model[k_component].append(cs_npmi_test)

            y_umass_model[k_component].append(cs_u_mass)
            test_y_umass_model[k_component].append(cs_u_mass_test)

            execution_times[k_component].append(k_components_time + graph_creation_time)

    save_model_scores(x_values=x, models=list(y_topics.keys()), model_topics=y_topics, model_c_v_scores=y_c_v_model,
                      model_npmi_scores=y_npmi_model, model_c_v_test_scores=test_y_c_v_model,
                      model_npmi_test_scores=test_y_npmi_model,
                      model_u_mass_scores=y_umass_model, model_u_mass_test_scores=test_y_umass_model,
                      execution_time=execution_times, number_of_nodes=number_of_nodes,
                      filename_prefix='k-components',
                      model_dbs_scores=y_dbs_model, x_label="Percentile Cutoff")

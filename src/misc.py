from sklearn.metrics.pairwise import cosine_similarity
import math
from src.visualizations import *


def absolute_word_counter(processed_docs: list) -> Counter:
    """
    absolute_word_counter creating a counter for the input data

    :param processed_docs:  list of preprocessed segments
    :return: Counter
    """
    temp_doc = []
    for doc in processed_docs:
        temp_doc.extend(doc)
    return Counter(temp_doc)


def get_word_weights(processed_docs: list, vocab: list, n_words: int, weight_type: str = 'tf') -> dict:
    """
    get_word_weights calculates the weights of all the vocabulary words

    :param processed_docs: list of preprocessed segments
    :param vocab: list of vocabulary words
    :param n_words: number of in data set
    :param weight_type: counting method

    :return: dictionary that maps each word in its weight
    """

    assert weight_type in ["tf", "tf-df", "tf-idf"], "weight_type not in ('tf', 'tf-df', 'tf-idf')"

    n_docs = len(processed_docs)
    absolute_counter = absolute_word_counter(processed_docs)

    word_weight = {}
    if weight_type == "tf":
        # calculate tf

        for w in vocab:
            word_weight.update({w: absolute_counter[w] / n_words})

    elif weight_type == "tf-df":
        # calculate tf-idf

        # calculate words frequencies per document
        word_frequencies_per_doc = [Counter(doc) for doc in processed_docs]

        # calculate document frequency
        words_per_doc = [list(word_frequency.keys()) for word_frequency in word_frequencies_per_doc]
        document_frequencies = Counter([w for doc in words_per_doc for w in doc])

        for w in vocab:
            df = document_frequencies[w] / n_docs
            tf = absolute_counter[w] / n_words
            word_weight.update({w: tf * df})

    else:
        assert weight_type == "tf-idf"
        # calculate tf-idf

        # calculate words frequencies per document
        word_frequencies_per_doc = [Counter(doc) for doc in processed_docs]

        # calculate document frequency
        words_per_doc = [list(word_frequency.keys()) for word_frequency in word_frequencies_per_doc]
        document_frequencies = Counter([w for doc in words_per_doc for w in doc])

        for w in vocab:
            idf = math.log(n_docs / (document_frequencies[w] + 1))
            tf = absolute_counter[w] / n_words
            word_weight.update({w: tf * idf})

    assert len(word_weight) == len(vocab)

    return word_weight


def get_nearest_indices(embedding, list_embedding, n_nearest: int = 10) -> list:
    """
    get_most_similar_indices finds the indices that are most similar to the input embedding

    :param embedding: a single embedding
    :param list_embedding: a list of embeddings, the indices
    :param n_nearest: number of indices return

    :return: list of the nearest indices
    """

    sim_matrix = cosine_similarity(embedding.reshape(1, -1), list_embedding)[0]
    most_sim = np.argsort(sim_matrix, axis=None)[:: -1]

    return most_sim[:n_nearest]


def save_model_scores(x_values: list, models: list, model_topics: dict, model_c_v_scores: dict, model_npmi_scores: dict,
                      model_c_v_test_scores: dict, model_npmi_test_scores: dict, model_u_mass_scores: dict,
                      model_u_mass_test_scores: dict, filename_prefix: str, execution_time: dict,
                      number_of_nodes: list,
                      model_dbs_scores: dict = None, x_label: str = "Number of Topics"):
    """
    save_model_scores documents the topic modeling performance

    :param x_values: list of x axis values
    :param models: list of different topic models used
    :param model_topics: list of list of topics
    :param model_c_v_scores: list of c_v scores for each topic model
    :param model_npmi_scores: list of NPMI scores for each topic model

    :param model_c_v_test_scores: list of c_v scores (extrinsic) for each topic model
    :param model_npmi_test_scores: list of NPMI scores (extrinsic) for each topic model

    :param model_u_mass_scores: list of u_mass scores for each topic model
    :param model_u_mass_test_scores: list of u_mass scores for each topic model

    :param filename_prefix: filename prefix

    :param execution_time: list of process times for each topic model
    :param number_of_nodes: list of number of nodes in each graph
    :param model_dbs_scores: list of DBS scores for each topic model

    :param x_label: name of the x-axis for the figures
    """

    # c_v coherence score figure - intrinsic
    ys = [temp for temp in model_c_v_scores.values()]
    _, fig = scatter_plot(x_values, ys, x_label=x_label, y_label="Coherence Score (c_v)",
                          color_legends=models, score_type='c_v')
    fig.savefig("visuals/" + filename_prefix + "_c_v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # NPMI coherence score figure - intrinsic
    ys = [temp for temp in model_npmi_scores.values()]
    _, fig = scatter_plot(x_values, ys, x_label=x_label, y_label="Coherence Score (NPMI)",
                          color_legends=models, score_type='c_npmi')
    fig.savefig("visuals/" + filename_prefix + "_c_npmi_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score figure - intrinsic
    ys = [temp for temp in model_u_mass_scores.values()]
    _, fig = scatter_plot(x_values, ys, x_label=x_label, y_label="Coherence Score (u_mass)",
                          color_legends=models, score_type='c_u_mass')
    fig.savefig("visuals/" + filename_prefix + "_c_uMass_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score figure - extrinsic
    ys = [temp for temp in model_u_mass_test_scores.values()]
    _, fig = scatter_plot(x_values, ys, x_label=x_label, y_label="Coherence Score (u_mass)",
                          color_legends=models, score_type='c_u_mass')
    fig.savefig("visuals/" + filename_prefix + "_extrinsic_c_uMass_vs_k.pdf", bbox_inches='tight', transparent=True)

    # c_v coherence score figure - extrinsic
    ys = [temp for temp in model_c_v_test_scores.values()]
    _, fig = scatter_plot(x_values, ys, x_label=x_label, y_label="Coherence Score (c_v)",
                          color_legends=models, score_type='c_v')
    fig.savefig("visuals/" + filename_prefix + "_extrinsic_c_v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # NPMI coherence score figure - extrinsic
    ys = [temp for temp in model_npmi_test_scores.values()]
    _, fig = scatter_plot(x_values, ys, x_label=x_label, y_label="Coherence Score (NPMI)",
                          color_legends=models, score_type='c_npmi')
    fig.savefig("visuals/" + filename_prefix + "_extrinsic_c_npmi_vs_k.pdf", bbox_inches='tight', transparent=True)

    # execution times
    ys = [temp for temp in execution_time.values()]
    _, fig = scatter_plot(x_values, ys, x_label=x_label, y_label="Seconds", color_legends=models, score_type='secs')
    fig.savefig("visuals/" + filename_prefix + "_execution_time.pdf", bbox_inches='tight', transparent=True)

    # number of nodes
    ys = number_of_nodes
    _, fig = scatter_plot(x_values, ys, x_label=x_label, y_label="Number of Nodes", score_type='nodes')
    fig.savefig("visuals/" + filename_prefix + "_number_of_nodes.pdf", bbox_inches='tight', transparent=True)

    # save all topics with their scores
    for m in models:

        if model_dbs_scores is not None:
            dbs_scores = model_dbs_scores[m]
        else:
            dbs_scores = None

        vis_topics_score(model_topics[m], model_c_v_scores[m], model_npmi_scores[m], model_c_v_test_scores[m],
                         model_npmi_test_scores[m], model_u_mass_scores[m], model_u_mass_test_scores[m],
                         execution_time[m], number_of_nodes, "visuals/clusters_eval_" + m.replace(" ", "-") + ".txt",
                         dbs_scores=dbs_scores)

    plt.close('all')

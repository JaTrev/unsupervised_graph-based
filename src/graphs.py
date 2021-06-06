import networkx as nx
from src.misc import *
import time

plt.rcParams['figure.figsize'] = [16, 9]


def remove_edges(graph: nx.Graph, edge_weights: list, percentile_cutoff: int, remove_isolated_nodes: bool = 1) \
        -> nx.Graph:
    """
    remove_edges removes edges from graph that have a weight below the weight cutoff

    :param graph: word embedding graph
    :param edge_weights: list or edge weights
    :param percentile_cutoff: cutoff weight percentile
    :param remove_isolated_nodes: if 1, remove isolated nodes (default: 1)

    :return: processed graph
    """
    # remove edges that do not have a high enough similarity score
    min_cutoff_value = np.percentile(edge_weights, percentile_cutoff)
    # min(heapq.nlargest(percentile_cutoff, edge_weights))

    graph_edge_weights = nx.get_edge_attributes(graph, "weight")

    edges_to_kill = []
    for edge in graph.edges():
        edge_weight = graph_edge_weights[edge]

        if edge_weight < min_cutoff_value:
            edges_to_kill.append(edge)

    for edge in edges_to_kill:
        graph.remove_edge(edge[0], edge[1])

    if remove_isolated_nodes:
        graph.remove_nodes_from(list(nx.isolates(graph)))

    return graph


def create_networkx_graph(words: list, word_embeddings: list, similarity_threshold: float = 0.4,
                          percentile_cutoff: int = 70, remove_isolated_nodes: bool = True,
                          method: str = "using_cutoff", top_n: int = 10) -> Tuple[nx.Graph, float]:
    """
    create_networkx_graph creates a graph given the words and their embeddings

    :param words: list of words which will be the nodes
    :param word_embeddings: embeddings of the words
    :param similarity_threshold: cosine similarity threshold value for the edges
    :param percentile_cutoff: percentile threshold value
    :param remove_isolated_nodes: boolean indicating if isolated nodes should be removed
    :param method: method used to trim the original graph
    :param top_n: is method is "using_top_n" then only the top_n highest weighted edges of each node is used

    :return:
        - word embedding graph
        - graph creation time
    """
    assert len(words) == len(word_embeddings), "words and word_embeddings must have the same length"
    assert method in ["using_top_n", "using_cutoff"]

    start_time = time.process_time()
    edge_weights = []

    # create undirected graph
    graph = nx.Graph()

    # split word embedding list in two halves
    first_half_length = int(len(word_embeddings) / 2)
    first_half = word_embeddings[:first_half_length]
    second_half = word_embeddings[first_half_length:]
    sim_matrix = cosine_similarity(first_half, second_half)

    for i in range(len(first_half)):

        # sort edges of node i by edge weight (similarity score)
        i_sim_vector = sim_matrix[i]
        sim_i_sorted_index = sorted(range(len(i_sim_vector)), key=i_sim_vector.__getitem__, reverse=True)

        # if method == "using_top_n":
        j_indices = sim_i_sorted_index[:10]
        # else: j_indices = sim_i_sorted_index

        # iterate over all relevant adjacent nodes
        for j in j_indices:
            sim = i_sim_vector[j]

            if method == "using_cutoff" and sim < similarity_threshold:
                break

            else:
                word_i = words[i]
                word_j = words[first_half_length + j]
                graph.add_edge(word_i, word_j, weight=float(sim))
                edge_weights.append(sim)

    graph_creation_time = time.process_time() - start_time

    if method == "using_top_n":
        return remove_edges(graph, edge_weights, 50, remove_isolated_nodes), graph_creation_time
    else:
        return remove_edges(graph, edge_weights, percentile_cutoff, remove_isolated_nodes), graph_creation_time


def sort_words_by(graph, word: str, word_weights: dict) -> Tuple[int, float, float]:
    """
    sort_words_by returns a tuple that is used to sort words with each topic calculated from k-components

    :param graph: word embedding graph
    :param word: list of words
    :param word_weights: list of word weights

    :return:
        - w_degree - degree of the node
        - sim_score - pooled similarity score of node
        - w_weight - word weight
    """

    neighbor_weights = []
    for w_neighbor in graph.adj[word]:
        neighbor_weights.append(float(graph.adj[word][w_neighbor]['weight']))

    sim_score = np.average(neighbor_weights)
    w_degree = graph.degree(word)
    w_weight = word_weights[word]

    return w_degree, sim_score, w_weight

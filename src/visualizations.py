import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from src.evaluation import *
import os
import numpy as np
from collections import Counter
from typing import Tuple


def vis_classification_score(topics_list: list, model_type: str, doc_labels_true: list, doc_topics_pred_list: list,
                             filename, n_words=10, multiple_true_label_set=False):
    """
    vis_classification_score creates a file and lists the topics and the classification performance of the topic model

    :param topics_list: list of topic lists
    :param model_type: topic model used in classification
    :param doc_labels_true: true labels of all segments
    :param doc_topics_pred_list: predicted labels of all segments
    :param filename: filename of the file created
    :param n_words: number of topic representatives (default: n_words = 10)
    :param multiple_true_label_set: flag used to state that doc_labels_true is a list of lists (default: False)

    """

    # filename = "visuals/classification_scores.txt"
    if any([len(t) > n_words for topics in topics_list for t in topics]):
        new_topics_list = [[t[:10] for t in topics] for topics in topics_list]
        topics_list = new_topics_list

    with open(filename, "w") as myFile:
        myFile.write('Model:  ' + str(model_type) + '\n')

        for i, topics in enumerate(topics_list):

            for i_t, t in enumerate(topics):
                myFile.write('Topic ' + str(i_t + 1) + '\n')

                for w in t:
                    myFile.write(str(w) + ' ')

                myFile.write('\n')

            myFile.write('\n')

            if multiple_true_label_set:
                true_labels = doc_labels_true[i]

            else:
                true_labels = doc_labels_true

            assert len(true_labels) == len(doc_topics_pred_list[i])
            myFile.write("ari score: " + ": " + str(ari_score(true_labels, doc_topics_pred_list[i])) + '\n')
            myFile.write("ami score: " + ": " + str(ami_score(true_labels, doc_topics_pred_list[i])) + '\n')
            myFile.write("nmi score: " + ": " + str(nmi_score(true_labels, doc_topics_pred_list[i])) + '\n')

            myFile.write('\n\n\n')


def label_distribution(doc_labels_true: list, doc_topics_pred: list, model_name: str):
    """
    label_distribution is used to visualize the distribution of the true labels of the segments
    classified to a predicted topic.

    :param doc_labels_true: list of true labels
    :param doc_topics_pred: list of predicted labels
    :param model_name: name of the model that predicted the labels

    """

    assert len(doc_labels_true) == len(doc_topics_pred), "labels must have same length"

    labels_true = np.array(doc_labels_true)
    labels_predicted = np.array(doc_topics_pred)
    parent_dir = "visuals/" + model_name + "_dir"
    os.mkdir(parent_dir)
    colors = ["#d14035", "#eb8a3c", "#ebb481", "#775845", "#31464f", "#86aa40", "#33655b", "#7ca2a1", "#B9EDF8",
              "#39BAE8"]

    topics = set(doc_topics_pred)
    assert -1 not in topics

    for t in range(len(topics)):
        predicted_indices = np.argwhere(labels_predicted == t)

        t_true = labels_true[predicted_indices].flatten()
        labels, values = zip(*Counter(t_true).items())
        values = list(values)
        max_values = int(max(values) / 20) * 20 + 40
        t_true_list = [[t_label for t_label in t_true if i == t_label] for i in range(10)]

        fig, ax = vis_prep()
        ax.set_xlabel("T$_{" + str(t+1) + "}$'s " + "True Topic Distribution", fontsize='medium', labelpad=4)
        ax.set_ylabel("Number of Segments", fontsize='medium', labelpad=4)
        ax.tick_params(axis='both', labelsize='small')
        plt.setp(ax.spines.values(), linewidth=2)
        plt.grid(color='grey', axis='y', linestyle='--', linewidth=0.7)

        plt.bar(range(len(t_true_list)), height=[len(labels) for labels in t_true_list], width=0.8,
                color=colors[:len(t_true_list)])

        plt.xticks(list(range(10)), list(range(1, 11)))
        ax.yaxis.set_ticks(list(range(0, max_values, 20)))

        fig.savefig(parent_dir + "/topic" + str(t+1) + ".pdf", bbox_inches='tight', transparent=True)

        # close fig
        plt.close(fig)


def vis_topics_score(topics_list: list, c_v_scores: list, npmi_scores: list, test_c_v_scores: list,
                     test_npmi_scores: list, u_mass_scores: list, test_u_mass_scores: list, execution_times: list,
                     number_of_nodes: list, filename: str, dbs_scores: list = None, n_words: int = 10):
    """
    vis_topics_score is used to a file that lists the resulting topics from a topic model and its performance scores

    :param topics_list: list of topics
    :param c_v_scores:  list of c_v coherence scores (intrinsic)
    :param npmi_scores: list of NPMI scores (intrinsic)
    :param test_c_v_scores: list of c_v coherence scores (extrinsic)
    :param test_npmi_scores: list of NPMI scores (extrinsic)
    :param u_mass_scores: list of u_mass scores
    :param test_u_mass_scores: list of u_mass scores (extrinsic)

    :param execution_times: list of process times
    :param number_of_nodes: list of number of nodes

    :param filename: name of the file
    :param dbs_scores: dbs score (intrinsic)
    :param n_words: number of topic representatives lists for each topic

    """
    assert len(topics_list) == len(c_v_scores)
    assert len(c_v_scores) == len(npmi_scores)

    if dbs_scores is not None:
        assert len(npmi_scores) == len(dbs_scores)

    if any([len(t) > n_words for topics in topics_list for t in topics]):
        new_topics_list = [[t[:10] for t in topics] for topics in topics_list]
        topics_list = new_topics_list

    with open(filename, "w") as myFile:

        for i, topics in enumerate(topics_list):

            for i_t, t in enumerate(topics):
                myFile.write('Topic ' + str(i_t + 1) + '\n')

                for w in t:
                    myFile.write(str(w) + ' ')

                myFile.write('\n')

            myFile.write("intrinsic evaluation" + '\n')
            myFile.write("c_v score: " + str(c_v_scores[i]) + '\n')
            myFile.write("u_mass score: " + str(u_mass_scores[i]) + '\n')
            myFile.write("npmi score: " + str(npmi_scores[i]) + '\n')

            myFile.write('\n')

            myFile.write("extrinsic evaluation" + '\n')
            myFile.write("c_v score: " + str(test_c_v_scores[i]) + '\n')
            myFile.write("u_mass score: " + str(test_u_mass_scores[i]) + '\n')
            myFile.write("npmi score: " + str(test_npmi_scores[i]) + '\n')

            if dbs_scores is not None:
                myFile.write('\n')
                myFile.write("dbs score: " + ": " + str(dbs_scores[i]) + '\n')

            myFile.write('\n')

            myFile.write('execution time: ' + "{:.6f}".format(execution_times[i]) + '\n')
            myFile.write('number of nodes: ' + str(number_of_nodes[i]) + '\n')

            myFile.write('\n\n\n')


def vis_prep() -> Tuple[plt.Figure, plt.Axes]:
    """
    vis_prep is used to set of the plt figures, including setting font size, axis spines, and colors

    :returns:
        - fig - plot figure
        - ax - plot axes
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis='both', labelsize=12)

    # mpl.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.size'] = 25
    plt.rcParams['axes.linewidth'] = 2

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')

    ax.yaxis.grid(color='grey', linestyle="--")
    ax.xaxis.grid(alpha=0)
    # plt.margins(0)

    return fig, ax


def scatter_plot(x: list, ys: list, x_label: str, y_label: str, score_type: str, color_legends: list = None) \
        -> (plt, plt.figure):
    """
    scatter_plot plots the x and y values in a figure.

    :param x: list of x values
    :param ys: list of y value lists
    :param x_label: x axis label
    :param y_label: y axis label
    :param score_type: score being plotted (u_mass, c_npmi, c_v, dbs, ari, ami, acc)
    :param color_legends: list of colors for each y list
    :return:
        - plt - figure plot
        - fig - resulting figure
    """
    fig, ax = vis_prep()

    assert score_type in ["c_v", "u_mass", "c_npmi", "c_u_mass", "dbs", "ari", "ami", "acc", "secs", "nodes"], \
        "wrong score_type of scatter plot, using: " + str(score_type)
    # assert isinstance(ys[0], list), "ys needs to be a list of list(s)"
    # assert len(ys) == len(color_legends), "need a color legend for each y list (ys)"

    error_value = -1000
    if color_legends is not None:
        mapper = mpl.cm.get_cmap('Pastel2')
        ys_color = [mapper(i_y) for i_y, _ in enumerate(ys)]

        for i_y, y in enumerate(ys):
            new_y = [value if value != error_value else np.nan for value in y]
            plt.plot(x, new_y, 'o-', c=ys_color[i_y], markersize=17, linewidth=3, label=color_legends[i_y])
    else:
        plt.plot(x, ys, 'o-', color='black', markersize=17, linewidth=3, )

    if score_type == "u_mass":
        y_ticks = [x for x in range(-6, 8, 2)]

    elif score_type == "c_npmi":
        y_ticks = [x/10 for x in range(-1, 6, 1)]

    elif score_type == "c_v":
        y_ticks = [x / 10 for x in range(0, 11, 1)]

    elif score_type == "c_u_mass":
        y_ticks = [x for x in range(-25, 5, 5)]

    elif score_type == "dbs":
        all_y = []
        for y in ys:
            all_y.extend(y)
        y_ticks = [x/10 for x in range(00, 40, 5)]

    elif score_type == "secs":
        y_ticks = [x/10 for x in range(0, 10, 2)]

    elif score_type == "nodes":
        y_ticks = [x for x in range(0, 300, 50)]

    else:
        assert score_type in ["ari", "ami", "acc"]
        y_ticks = [x / 10 for x in range(0, 7, 1)]

    ax.set_xlabel(x_label, fontsize='medium', labelpad=4)
    ax.set_ylabel(y_label, fontsize='medium', labelpad=4)

    ax.yaxis.set_ticks(y_ticks)
    ax.xaxis.set_ticks(x)

    ax.tick_params(axis='both', labelsize='small')

    if color_legends is not None:
        plt.legend(fontsize='x-small')
    plt.setp(ax.spines.values(), linewidth=2)
    plt.grid(color='grey', axis='y', linestyle='--', linewidth=0.7)
    return plt, fig


def number_of_words_per_doc(raw_segment_data: list):
    """
    number_of_words_per_doc calculates a bar chart for the segment lengths
    :param raw_segment_data: list of raw segments
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    mpl.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.size'] = 25
    plt.rcParams['axes.linewidth'] = 2

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')

    ax.yaxis.grid(color='grey', linestyle="--")
    ax.xaxis.grid(alpha=0)

    plt.margins(0)

    all_data_lengths = [len([w for w in doc.split() if w.isalpha()]) for doc in raw_segment_data]
    data_lengths_c = [all_data_lengths.count(int(i)) for i in range(int(np.max(all_data_lengths)))]
    plt.bar(range(int(np.max(all_data_lengths))), data_lengths_c, color="black")

    plt.xlim(right=int(np.max(all_data_lengths)))
    plt.xlim(left=0)

    plt.ylim(top=int(np.max(data_lengths_c)))
    plt.ylim(bottom=0)

    ax.set_xlabel("Number of Words", fontsize="medium")
    ax.set_ylabel("Number of Segments", fontsize="medium")

    plt.show()
    fig.savefig("visuals/segment_word_distribution.pdf", bbox_inches='tight', transparent=True)
    plt.close(fig)


def vis_most_common_words(data: list, raw_data: False, preprocessed: False):
    """
    vis_most_common_words produces a bar chart with of the most common words

    :param data: list of segments
    :param raw_data: list of raw segments
    :param preprocessed: flag for preprocessing

    """
    if raw_data:
        data = [doc.split() for doc in data]
        y_max = 25000
        filename = "most_common_words"
    else:
        if preprocessed:

            y_max = 1000
        else:

            y_max = 4000
        filename = "processed_most_common_words"

    fig, ax = plt.subplots(figsize=(10, 6))

    mpl.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.linewidth'] = 2

    ax.tick_params(axis='both', labelsize=12)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')

    ax.yaxis.grid(color='grey', linestyle="--", alpha=0.5)
    ax.xaxis.grid(alpha=0)
    plt.margins(0)

    data_words = []
    for doc in data:
        data_words.extend([w.lower() for w in doc if w.isalpha()])

    data_words_c = Counter(data_words)

    most_common_words = [w for w, c in data_words_c.most_common(30)]
    most_common_words_c = [c for w, c in data_words_c.most_common(30)]

    plt.bar(most_common_words, most_common_words_c, color='black', width=0.5)

    plt.ylim(top=y_max)
    plt.ylim(bottom=0)

    ax.set_xlabel("Top 30 Words", fontsize="medium")
    ax.set_ylabel("Number of Occurrences", fontsize="medium")

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")

    fig.savefig("visuals/" + str(filename) + ".pdf", bbox_inches='tight', transparent=True)
    plt.close(fig)

from src.get_data import *
from src.preprocessing import *
from src.doc_space_topicModeling import *
from src.jointly_embedded_space import *
from src.word_space_topicModeling import *
import argparse


def main_fct(preprocessed_data_set: str, topic_model_type: str, misc_prints: str, raw_data: list, raw_labels: list,
             bert_embedding_type: str = "normal_ll",
             raw_test_data: list = None, raw_test_labels: list = None):
    """
    main function performing all topic models

    :param preprocessed_data_set: name of preprocessed data set
    :param topic_model_type: name of topic model
    :param misc_prints: misc. print functions
    :param raw_data: original segments
    :param raw_labels: original topics of the segments
    :param raw_test_data: original test set
    :param raw_test_labels: original test set labels
    :param bert_embedding_type: BERT embedding type
    """

    assert preprocessed_data_set in ["JN", "FP"]
    assert (topic_model_type in ["Baseline", "RRW", "TVS", "k-components", "BERT", "avg_w2v", "doc2vec"]
            or misc_prints in ["segment_size", "common_words"])

    if preprocessed_data_set == "JN":
        do_lemmatizing = False
        do_stop_word_removal = False
    else:
        do_lemmatizing = True
        do_stop_word_removal = True

    data_processed, data_processed_labels, vocab, tokenized_docs = preprocessing(
        raw_data, raw_labels, do_lemmatizing=do_lemmatizing, do_stop_word_removal=do_stop_word_removal)

    if (raw_test_data is not None) and (raw_test_labels is not None):
        _, _, _, test_tokenized_docs = preprocessing(raw_test_data, raw_test_labels, do_lemmatizing=do_lemmatizing,
                                                     do_stop_word_removal=do_stop_word_removal)
    else:
        test_tokenized_docs = None

    # perform topi modeling based on topic_model_type
    if topic_model_type == "Baseline":
        baseline_topic_model(data_processed, vocab, tokenized_docs, data_processed_labels, test_tokenized_docs)

    elif topic_model_type == "RRW":
        word2vec_topic_model(data_processed, vocab, tokenized_docs, test_tokenized_docs,
                             data_set_name=preprocessed_data_set, topic_vector_flag=False)

    elif topic_model_type == "TVS":
        word2vec_topic_model(data_processed, vocab, tokenized_docs, test_tokenized_docs,
                             data_set_name=preprocessed_data_set, topic_vector_flag=True)

    elif topic_model_type == "k-components":
        k_components_model(data_processed, vocab, tokenized_docs, test_tokenized_docs,
                           data_set_name=preprocessed_data_set)

    elif topic_model_type == "BERT":
        bert_topic_model(bert_embedding_type, data_processed, vocab, test_tokenized_docs)

    elif topic_model_type == "avg_w2v":
        w_d_clustering(data_processed, preprocessed_data_set, vocab, data_processed_labels,
                       test_tokenized_docs, segment_embedding_type="w2v_avg", true_topic_amount=10)

    elif topic_model_type == "doc2vec":
        w_d_clustering(data_processed, preprocessed_data_set, vocab, data_processed_labels,
                       test_tokenized_docs, segment_embedding_type="doc2vec", true_topic_amount=10)
    else:
        # topic_model_type == 'null'
        if misc_prints == "segment_size":
            temp_segments = [seg for seg in new_data]
            temp_segments.extend(new_test_data)
            number_of_words_per_doc(temp_segments)

        else:
            assert misc_prints == "common_words", "must define a valid topic_model_type or a misc_prints"
            vis_most_common_words(data_processed, raw_data=False, preprocessed=True)


if __name__ == "__main__":

    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--pds', dest='data_set', type=str, required=True,  help="state the preprocessed data set should be used: Just Nouns (JN), Fully Preprocessed (FP)")

    parser.add_argument('--tm', dest='topic_model', type=str, required=True,
                        help="state what topic model that should be used")

    parser.add_argument('--mp', dest='misc_prints', type=str, required=False, default=None,
                        help="define the miscellaneous print that should be performed")

    parser.add_argument('--bert', dest='bert_type', type=str, required=False, default="normal_ll",
                        help="define the BERT embedding type")

    parser.add_argument('--do_testing', dest='do_testing', required=False, default=True, action='store_true',
                        help="should test data set be used")
    args = parser.parse_args()

    assert args.data_set in ["JN", "FP"], "need to select a proper preprocessing schema [JN, FP]"
    assert args.topic_model in ['Baseline', 'RRW', 'TVS', 'k-components', 'BERT', 'avg_w2v', 'doc2vec', 'null'], (
        "select one of the topic models: ['Baseline', 'RRW', 'TVS', 'k-components', 'BERT', 'avg_w2v', 'doc2vec', "
        "'null']")

    if args.topic_model == "null":
        assert args.misc_prints in ['segment_size', 'common_words'], ("select one of the misc_prints:  "
                                                                      "['segment_size', 'common_words']")
    print("do_test: " + str(args.do_testing))
    # check if testing you be performed
    if args.do_testing:
        data, data_labels, test_data, test_data_labels = get_data()

        new_test_data = []
        new_test_data_label = []
        for i, d in enumerate(test_data):

            # remove very short test segments
            if len([w for w in d.split() if w.isalpha()]) > 2:
                new_test_data.append(d)
                new_test_data_label.append(test_data_labels[i])
    else:

        with open("data/saved_data.pickle", "rb") as myFile:
            data = pickle.load(myFile)

        with open("data/saved_data_labels.pickle", "rb") as myFile:
            data_labels = pickle.load(myFile)

    new_data = []
    new_data_label = []
    for i, d in enumerate(data):

        # remove segments very short segments
        if len([w for w in d.split() if w.isalpha()]) > 2:
            new_data.append(d)
            new_data_label.append(data_labels[i])

    print("len of data: " + str(len(new_data)))

    main_fct(preprocessed_data_set=args.data_set, topic_model_type=args.topic_model, misc_prints=args.misc_prints,
             bert_embedding_type=args.bert_type, raw_data=new_data, raw_labels=new_data_label,
             raw_test_data=new_test_data, raw_test_labels=new_test_data_label)

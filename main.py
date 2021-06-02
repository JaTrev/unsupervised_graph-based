from src.get_data import *
from src.preprocessing import *
from src.topic_modeling import *
import argparse


def main_fct(preprocessed_data_set: str, topic_model_type: str, raw_data: list, raw_labels: list,
             raw_test_data: list = None, raw_test_labels: list = None):
    """
    main function performing all topic models

    :param preprocessed_data_set: name of preprocessed data set
    :param topic_model_type: name of topic model
    :param raw_data: original segments
    :param raw_labels: original topics of the segments
    :param raw_test_data: original test set
    :param raw_test_labels: original test set labels
    """

    data_processed, data_processed_labels, vocab, tokenized_docs = preprocessing(
        raw_data, raw_labels, preprocessing_type=preprocessed_data_set)

    if (raw_test_data is not None) and (raw_test_labels is not None):
        _, _, _, test_tokenized_docs = preprocessing(raw_test_data, raw_test_labels,
                                                     preprocessing_type=preprocessed_data_set)
    else:
        test_tokenized_docs = None

    assert 0
    # perform topi modeling based on topic_model_type
    if topic_model_type == "RRW":
        word2vec_topic_model(data_processed, vocab, tokenized_docs, test_tokenized_docs,
                             data_set_name=preprocessed_data_set, topic_vector_flag=False)

    elif topic_model_type == "TVS":
        word2vec_topic_model(data_processed, vocab, tokenized_docs, test_tokenized_docs,
                             data_set_name=preprocessed_data_set, topic_vector_flag=True)

    else:
        assert topic_model_type == "k-components"
        k_components_model(data_processed, vocab, tokenized_docs, test_tokenized_docs,
                           data_set_name=preprocessed_data_set)


if __name__ == "__main__":

    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--pds', dest='data_set', type=str, required=True,
                        help="state the preprocessed data set should be used: Just Nouns (JN), Fully Preprocessed (FP)")

    parser.add_argument('--tm', dest='topic_model', type=str, required=True,
                        help="state what topic model that should be used")

    parser.add_argument('--do_testing', dest='do_testing', required=False, default=False, action='store_true',
                        help="should test data set be used")
    args = parser.parse_args()

    assert args.data_set in ["JN", "FP"], "need to select a proper preprocessing schema [JN, FP]"
    assert args.topic_model in ['RRW', 'TVS', 'k-components'], (
        "select one of the topic models: ['RRW', 'TVS', 'k-components']")

    print("do_test: " + str(args.do_testing))

    filtered_data, filtered_data_labels, filtered_test_data, filtered_test_data_labels = get_data(
        get_test_data=args.do_testing)

    main_fct(preprocessed_data_set=args.data_set, topic_model_type=args.topic_model,
             raw_data=filtered_data, raw_labels=filtered_data_labels,
             raw_test_data=filtered_test_data, raw_test_labels=filtered_test_data_labels)

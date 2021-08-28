from gensim.models import Word2Vec
import numpy as np
from typing import Tuple
from pathlib import Path
import pickle


def create_w2v_model(processed_data: list, min_c: int, win: int, negative: int, seed: int,
                     sample: float = 6e-5, alpha: float = 0.03, min_alpha: float = 0.0007, ns_exponent: float = 0.75,
                     epochs: int = 30, size: int = 300, sg=0, cbow_mean=1, hs=0) -> Word2Vec:
    """
    create_w2v_model create the Word2Vec using the specified parameters

    :param processed_data: preprocess data set
    :param min_c: minimum absolute frequency of a word
    :param win: maximum distance between the current and the predicted word
    :param negative: specifies how many "noise words" will be used
    :param seed: seed for the algorithm
    :param sample: threshold for downsampling
    :param alpha: initial learning rate
    :param min_alpha: learning rate drops in min_alpha during training
    :param ns_exponent: exponent used in negative sampling
    :param epochs: number of epochs
    :param size: embedding size
    :param sg: flag for skip-gram
    :param cbow_mean: flag for using mean in cbow architecture
    :param hs: flag for hierarchical softmax

    :return: trained Word2Vec model
    """

    w2v_model = Word2Vec(min_count=min_c,
                         window=win,
                         vector_size=size,
                         sample=sample,
                         alpha=alpha,
                         min_alpha=min_alpha,
                         negative=negative,
                         ns_exponent=ns_exponent,
                         sorted_vocab=1,
                         seed=seed,
                         compute_loss=True,
                         workers=1,
                         sg=sg,
                         cbow_mean=cbow_mean,
                         hs=hs
                         )
    w2v_model.build_vocab(processed_data, progress_per=10000)
    w2v_model.train(processed_data,
                    total_examples=w2v_model.corpus_count,
                    epochs=epochs, report_delay=1)
    # normalize vectors:
    w2v_model.init_sims(replace=True)

    return w2v_model


def get_word_vectors(processed_data: list, vocab: list, model_file_name: str, params: dict, data_set_name: str) -> \
        Tuple[list, list, Word2Vec]:
    """
    get_word_vectors calculates the word embeddings

    :param processed_data: list of processed documents
    :param vocab: list of words in the processed documents
    :param model_file_name: name of a previously saved Word2Vec model
    :param params: parameters for a Word2Vec
    :param data_set_name: name of data set being examined

    :rtype: list, list, Word2Vec
    :return:
        - vocab_words - list of vocabulary words
        - vocab_embeddings - list of embeddings for the vocabulary words
        - w2v_model - Word2Vec model
    """

    if data_set_name == "CRR":
        # fetch precalculated word2vec model
        with open("data/restaurant/w2v_model.pickle", "rb") as myFile:
            w2v_model = pickle.load(myFile)

        with open("data/restaurant/vocab_words.pickle", "rb") as myFile:
            vocab_words = pickle.load(myFile)

        with open("data/restaurant/vocab_embeddings.pickle", "rb") as myFile:
            vocab_embeddings = pickle.load(myFile)

    else:
        if Path("data/" + model_file_name).is_file():

            with open("data/" + model_file_name, "rb") as myFile:
                w2v_model = pickle.load(myFile)

        else:

            assert isinstance(params, dict), "missing w2v_model params"

            assert {'min_c', 'win', 'negative', 'seed'}.issubset(params.keys()), (
                "missing w2v_model params, need: min_c', 'win', 'negative',', 'seed'")

            w2v_model = create_w2v_model(processed_data, **params)
            with open("data/" + model_file_name, "wb") as myFile:
                pickle.dump(w2v_model, myFile)

        # vocab_words and vocab_embeddings are sorted like vocab
        vocab_words = [w for w in vocab if w in w2v_model.wv.index_to_key]
        vocab_embeddings = [w2v_model.wv.vectors[w2v_model.wv.key_to_index[w]]
                            for w in vocab_words]
    return vocab_words, vocab_embeddings, w2v_model


def get_vocabulary_embeddings(training_data_processed: list, vocab: list, topic_model: str, model_file_name: str,
                              data_set_name: str) -> Tuple[list, list, Word2Vec]:
    """
    get_vocabulary_embeddings fetches the word embeddings of all relevant vocabulary words

    :param training_data_processed: training set
    :param vocab: list of vocabulary words, calculated in preprocessing
    :param topic_model: name of the topic modelling approach
    :param model_file_name: name of the saved topic model
    :param data_set_name: name of the preprocessed data set used
    :return:
        - vocab_words - list of vocabulary words
        - vocab_embeddings - list of embeddings for the vocabulary words
        - w2v_model - Word2Vec model
    """
    w2v_params_k_components = {"min_c": 50, "win": 15, "negative": 0, "sample": 1e-5,
                               "hs": 1, "epochs": 400, "sg": 1, 'seed': 42}
    w2v_params_baseline = {"min_c": 10, "win": 7, "negative": 0, "sample": 1e-5,
                           "hs": 1, "epochs": 400, "sg": 1, 'seed': 42,
                           'ns_exponent': 0.75}
    if topic_model == "k-components":
        w2v_params = w2v_params_k_components
    else:
        assert topic_model == "baseline"
        w2v_params = w2v_params_baseline

    return get_word_vectors(training_data_processed, vocab, model_file_name=model_file_name, params=w2v_params,
                            data_set_name=data_set_name)


def get_topic_vector(topic_embeddings: list) -> np.ndarray:
    """
    get_topic_vector calculates the topic vector from a list of embeddings

    :param topic_embeddings: list of embeddings
    :return: topic vector
    """

    return np.average(topic_embeddings, axis=0)

from gensim.models import Word2Vec
import numpy as np
from typing import Tuple


def get_saved_w2v_model(w2v_model: str) -> Word2Vec:
    """
    get_saved_w2v_model returns a pretrained Word2Vec model

    :param w2v_model: name of the pretrained model
    :return:  word2vec model
    """
    w2v_model = Word2Vec.load(w2v_model)
    return w2v_model


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


def get_word_vectors(processed_data: list, vocab: list, saved_model=None, params=None) -> Tuple[list, list, Word2Vec]:
    """
    get_word_vectors calculates the word embeddings

    :param processed_data: list of processed documents
    :param vocab: list of words in the processed documents
    :param saved_model: name of a previously saved Word2Vec model
    :param params: parameters for a Word2Vec

    :rtype: list, list, Word2Vec
    :return:
        - vocab_words - list of vocabulary words
        - vocab_embeddings - list of embeddings for the vocabulary words
        - w2v_model - Word2Vec model
    """

    if isinstance(saved_model, str):
        w2v_model = get_saved_w2v_model(saved_model)

    else:

        assert isinstance(params, dict), "missing w2v_model params"

        assert {'min_c', 'win', 'negative', 'seed'}.issubset(params.keys()), (
            "missing w2v_model params, need: min_c', 'win', 'negative',', 'seed'")

        w2v_model = create_w2v_model(processed_data, **params)

    # vocab_words and vocab_embeddings are sorted like vocab
    vocab_words = [w for w in vocab if w in w2v_model.wv.index_to_key]
    vocab_embeddings = [w2v_model.wv.vectors[w2v_model.wv.key_to_index[w]]
                        for w in vocab_words]

    return vocab_words, vocab_embeddings, w2v_model


def get_topic_vector(topic_embeddings: list) -> np.ndarray:
    """
    get_topic_vector calculates the topic vector from a list of embeddings

    :param topic_embeddings: list of embeddings
    :return: topic vector
    """

    return np.average(topic_embeddings, axis=0)

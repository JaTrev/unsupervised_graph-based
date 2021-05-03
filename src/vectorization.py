from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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
    :param min_alpha: learning rate drops in min_alpha during trainging
    :param ns_exponent: expoonent used in negative sampling
    :param epochs: number of epochs
    :param size: embedding size
    :param sg: flag for skip-gram
    :param cbow_mean: flag for using mean in cbow architecture
    :param hs: flag for hierarchical softmax

    :return: trained Word2Vec model
    """

    w2v_model = Word2Vec(min_count=min_c,
                         window=win,
                         size=size,
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
    vocab_words = [w for w in vocab if w in w2v_model.wv.index2word]
    vocab_embeddings = [w2v_model.wv.vectors[w2v_model.wv.index2word.index(w)]
                        for w in vocab_words]

    return vocab_words, vocab_embeddings, w2v_model


def semantic_space_embeddings(processed_data: list, data_labels: list, vocab: list, embedding_type: str,
                              params=None, saved_model=None) -> Tuple[list, list, list, list, list]:
    """
    semantic_space_embeddings calculates the document embeddings and word embeddings to create semantic space

    :param processed_data: preprocessed data set
    :param data_labels: list of true segment labels
    :param vocab: list of preprocessed vocabulary words
    :param embedding_type: document embedding type ("w2v_avg", "w2v_sum", "doc2vec")
    :param params: embedding model parameters used for training
    :param saved_model: name of saved trained model
    """
    assert len(processed_data) == len(data_labels)

    doc_embeddings = []

    if embedding_type == "w2v_avg":
        vocab_words, vocab_embeddings, _ = get_word_vectors(processed_data=processed_data, vocab=vocab,
                                                            saved_model=saved_model,  params=params)

        doc_data = []
        doc_labels = []
        for i, doc in enumerate(processed_data):
            if any([w in doc for w in vocab_words]):
                doc_data.append(doc)
                doc_labels.append(data_labels[i])

        for i, doc in enumerate(doc_data):
            temp_embeddings = [vocab_embeddings[vocab_words.index(w)] for w in doc if w in vocab_words]

            if len(temp_embeddings) > 1:
                doc_embeddings.append(np.mean(temp_embeddings, axis=0))

            elif len(temp_embeddings) == 1:
                doc_embeddings.append(temp_embeddings[0])
            else:
                print("error")
                print([w for w in doc if w in vocab_words])
                print("---------")
                continue

    elif embedding_type == "w2v_sum":
        vocab_words, vocab_embeddings, _ = get_word_vectors(processed_data=processed_data, vocab=vocab, params=params)

        doc_data = []
        doc_labels = []
        for i, doc in enumerate(processed_data):
            if any([w in doc for w in vocab_words]):
                doc_data.append(doc)
                doc_labels.append(data_labels[i])

        for i, doc in enumerate(doc_data):
            temp_embeddings = [vocab_embeddings[vocab_words.index(w)] for w in doc if w in vocab_words]

            if len(temp_embeddings) > 1:
                doc_embeddings.append(np.sum(temp_embeddings, axis=0))

            elif len(temp_embeddings) == 1:
                doc_embeddings.append(temp_embeddings[0])
            else:
                print("error")
                print([w for w in doc if w in vocab_words])
                print("---------")
                continue

    else:
        assert embedding_type == "doc2vec"

        doc_data = []
        doc_labels = []
        for i, doc in enumerate(processed_data):
            if any([w in doc for w in vocab]):
                doc_data.append(doc)
                doc_labels.append(data_labels[i])
        vocab_words, vocab_embeddings, doc_embeddings, _ = get_doc2vec_embeddings(doc_data, vocab, **params)

    assert all([len(doc_embeddings[0]) == len(e) for e in doc_embeddings])
    assert len(doc_data) == len(doc_embeddings)
    assert len(doc_data) == len(doc_labels)
    return doc_data, doc_labels, doc_embeddings, vocab_words, vocab_embeddings


def get_doc2vec_embeddings(processed_data: list, vocab: list, min_c: int, win: int, negative: int, hs: int,
                           seed: int, sample: float = 6e-5, alpha: float = 0.03, min_alpha: float = 0.0007,
                           epochs: int = 30, size: int = 300, ns_exponent: float = 0.75, dm=1, dbow_words=0) \
        -> Tuple[list, list, list, Doc2Vec]:

    """
    get_doc2vec_embeddings trains the Doc2Vec model on the given data and returns the embeddings
    :param processed_data: preprocessed data set
    :param vocab: vocabulary of the preprocessed data set
    :param min_c: minimum count threshold used by Doc2Vec model
    :param win: window size used
    :param negative: specifies how many "noise words" will be used
    :param hs: flag for hierarchical softmax
    :param seed: seed for the algorithm
    :param sample: threshold for downsampling
    :param alpha: initial learning rate
    :param min_alpha: learning rate drops in min_alpha during training
    :param epochs: number of training epochs
    :param size: embedding size
    :param ns_exponent: expoonent used in negative sampling
    :param dm: flag for PV-DM model
    :param dbow_words: flag for training word-vectors simultaneously with DBOW doc-vector training

    :returns:
        - vocab_words - list of vocabulary words
        - vocab_embeddings - list of vocabulary embeddings
        - doc_embeddings - list of document embeddings
        - Doc2Vec - trained Doc2Vec model
    """

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(processed_data)]
    d2v_model = Doc2Vec(documents, min_count=min_c, window=win, vector_size=size, sample=sample, alpha=alpha,
                        min_alpha=min_alpha, negative=negative, ns_exponent=ns_exponent, seed=seed, compute_loss=True,
                        workers=1, epochs=epochs, sorted_vocab=1, hs=hs, dm=dm, dbow_words=dbow_words)

    # normalize vectors:
    d2v_model.init_sims(replace=True)

    # vocab_words and vocab_embeddings are sorted like vocab
    vocab_words = [w for w in vocab if w in d2v_model.wv.index2word]
    vocab_embeddings = [d2v_model.wv.vectors[d2v_model.wv.index2word.index(w)] for w in vocab_words]
    doc_embeddings = [d2v_model.docvecs[i] for i in range(len(processed_data))]

    return vocab_words, vocab_embeddings, doc_embeddings, d2v_model


def get_topic_vector(topic_embeddings: list) -> np.ndarray:
    """
    get_topic_vector caluclates the topic vector from a list of embeddings

    :param topic_embeddings: list of embeddings
    :return: topic vector
    """

    return np.average(topic_embeddings, axis=0)

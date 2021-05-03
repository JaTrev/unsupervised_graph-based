from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import LdaModel
from gensim import corpora
from sklearn.decomposition import NMF
from typing import Tuple


def lda_topics(processed_data: list, n_topics: int = 10, learning_decay: float = 0.5,
               learning_offset: float = 1.0, max_iter: int = 50, n_words: int = 10) -> Tuple[list, list]:
    """
    lda_topics perfoms LDA topic modeling on the input data

    :param processed_data: list of preprocessed segments
    :param n_topics: number of topics to extract form corpus
    :param learning_decay: learning decay parameter for LDA
    :param learning_offset: learning offset parameter for LDA
    :param max_iter: max. number of interations
    :param n_words: number of topic representatives

    :return:
        - topics - list of topics (and their representatives
        - doc_topics - list of predicted topics, one for each segment
    """

    dictionary = corpora.Dictionary(processed_data, )
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_data]

    lda_model = LdaModel(doc_term_matrix, id2word=dictionary, num_topics=n_topics, offset=learning_offset,
                         random_state=42, update_every=1, iterations=max_iter,
                         passes=10, alpha='auto', eta="auto", decay=learning_decay, per_word_topics=True)

    topics = []
    for i_t, topic_word_dist in enumerate(lda_model.get_topics()):
        topic = [lda_model.id2word[w_id] for w_id, _ in lda_model.get_topic_terms(i_t, topn=n_words)]
        topics.append(topic)

    # getting documents topic labels
    doc_topics = []
    for doc in doc_term_matrix:

        doc_t_dist = sorted(lda_model.get_document_topics(doc), key=lambda item: item[1], reverse=True)
        t, _ = doc_t_dist[0]
        doc_topics.append(t)

    assert len(doc_topics) == len(processed_data)
    return topics, doc_topics


def nmf_topics(preprocessed_data: list, vocabulary: list, n_topics: int = 10,
               n_words: int = 10, init: str = 'nndsvd', solver: str = 'cd', beta_loss: str = 'frobenius',
               use_tfidf: bool = True) -> Tuple[list, list]:
    """

    :param preprocessed_data: list of preprocessed segments
    :param vocabulary: vocabulary words from the preprocessed segments
    :param n_topics: number of topics to extract form corpus
    :param n_words: number of topic representatives
    :param init: init method
    :param solver: numeric solver used by NMF
    :param beta_loss: divergence to be minized
    :param use_tfidf: if 1, using TF-IDF vectorization, if 0, using TF vectorization

    :return:
        - topics - list of topics (and their representatives
        - doc_topics - list of predicted topics, one for each segment
    """

    assert init in [None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar'], "need an appropriate method to init to procedure"
    assert solver in ['mu', 'cd'], "need an appropriate solver"
    assert beta_loss in ['frobenius', 'kullback-leibler', 'itakura-saito'], "need an appropriate beta_loss"
    str_split = " "
    raw_docs = [str_split.join(doc) for doc in preprocessed_data]

    if use_tfidf:
        tfidf_vectorizer = TfidfVectorizer(vocabulary=vocabulary,
                                           stop_words='english')
        new_data = tfidf_vectorizer.fit_transform(raw_docs)
        feature_names = tfidf_vectorizer.get_feature_names()

    else:
        tf_vectorizer = CountVectorizer(vocabulary=vocabulary,
                                        stop_words='english')
        new_data = tf_vectorizer.fit_transform(raw_docs)
        feature_names = tf_vectorizer.get_feature_names()

    nmf_model = NMF(n_components=n_topics, init=init, beta_loss=beta_loss, solver=solver,
                    max_iter=1000, alpha=.1, l1_ratio=.5, random_state=42)

    w_matrix = nmf_model.fit_transform(new_data)
    h_matrix = nmf_model.components_

    topics = []
    for topic_idx, topic in enumerate(h_matrix):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_words - 1: -1]]
        topics.append(top_features)

    doc_topics = []
    for w_doc in w_matrix:
        topic_dict = w_doc.argsort()[::-1]
        doc_topics.append(topic_dict[0])

    return topics, doc_topics

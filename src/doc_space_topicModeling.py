from src.model import *
from src.visualizations import *
from src.misc import save_model_scores


def baseline_topic_model(data_processed: list, vocab: list, tokenized_segments: list, seg_labels_true: list,
                         test_tokenized_segments: list, x: list = None):

    """
    baseline_topic_model performs tradtional baselin topic modeling and measures its performance

    :param data_processed: list of preprocessed segments
    :param vocab: list of vocabulary words
    :param tokenized_segments: original data set tokenized
    :param seg_labels_true: list of true segment labels
    :param test_tokenized_segments: list of test segments
    :param x: list of number of topics

    """

    # create x-values (number of topics list)
    if x is None:
        x = list(range(2, 22, 2))
    else:
        assert isinstance(x, list), "x has to be a list to iterate over"

    true_topic_amount = len(set(seg_labels_true))

    y_topics = {'NMF TF': [], 'NMF TF-IDF': [], 'LDA': []}
    y_c_v_model = {'NMF TF': [], 'NMF TF-IDF': [], 'LDA': []}
    y_npmi_model = {'NMF TF': [], 'NMF TF-IDF': [], 'LDA': []}
    test_y_c_v_model = {'NMF TF': [], 'NMF TF-IDF': [], 'LDA': []}
    test_y_npmi_model = {'NMF TF': [], 'NMF TF-IDF': [], 'LDA': []}
    doc_topics_pred_model = {'NMF TF': [], 'NMF TF-IDF': [], 'LDA': []}

    # iterate over x values (number of topics)
    for k in x:

        # iterate over all baseline topic models
        for m in list(y_topics.keys()):

            if m == 'NMF TF':
                topics, doc_topics_pred = nmf_topics(data_processed, vocabulary=vocab, n_topics=k, solver='cd',
                                                     beta_loss='frobenius', use_tfidf=False, n_words=50)

            elif m == 'NMF TF-IDF':
                topics, doc_topics_pred = nmf_topics(data_processed, vocabulary=vocab, n_topics=k, solver='cd',
                                                     beta_loss='frobenius', use_tfidf=True, n_words=50)
            else:
                assert m == 'LDA'
                topics, doc_topics_pred = lda_topics(data_processed, n_topics=k, n_words=50)

            y_topics[m].append(topics)

            # topic evaluation scores
            # intrinsic scores
            y_c_v_model[m].append(c_v_coherence_score(tokenized_segments, topics, cs_type='c_v'))
            y_npmi_model[m].append(npmi_coherence_score(tokenized_segments, topics, len(topics)))

            if test_tokenized_segments is not None:
                # extrinsic scores
                test_y_c_v_model[m].append(c_v_coherence_score(test_tokenized_segments, topics, cs_type='c_v'))
                test_y_npmi_model[m].append(npmi_coherence_score(test_tokenized_segments, topics, len(topics)))
            else:
                test_y_c_v_model[m].append(-1000.0)
                test_y_npmi_model[m].append(-1000.0)

            # save predicted topics for classification evaluation
            doc_topics_pred_model[m].append(doc_topics_pred)

            if k == true_topic_amount:
                label_distribution(seg_labels_true, doc_topics_pred, m.replace(" ", "-"))

    save_model_scores(x_values=x, models=list(y_topics.keys()), model_topics=y_topics, model_c_v_scores=y_c_v_model,
                      model_npmi_scores=y_npmi_model, model_c_v_test_scores=test_y_c_v_model,
                      model_npmi_test_scores=test_y_npmi_model, filename_prefix='BL')

    for m in list(y_topics.keys()):

        # save classification scores for every model
        vis_classification_score(y_topics[m], m, seg_labels_true, doc_topics_pred_model[m],
                                 filename="visuals/classification_scores_" + str(m) + ".txt",
                                 multiple_true_label_set=False)

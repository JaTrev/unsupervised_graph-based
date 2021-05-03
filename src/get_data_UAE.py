import codecs
import re
import operator
from src.preprocessing import *
from src.doc_space_topicModeling import *
from src.jointly_embedded_space import *
from src.word_space_topicModeling import *

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')


def is_number(token):
    return bool(num_regex.match(token))


def create_vocab(domain, maxlen=0, vocab_size=0):
    assert domain in {'restaurant', 'beer'}
    source = 'data/' + domain + '/train.txt'

    total_words, unique_words = 0, 0
    word_freqs = {}
    top = 0

    fin = codecs.open(source, 'r', 'utf-8')
    for line in fin:
        words = line.split()
        if maxlen > 0 and len(words) > maxlen:
            continue

        for w in words:
            if not is_number(w):
                try:
                    word_freqs[w] += 1
                except KeyError:
                    unique_words += 1
                    word_freqs[w] = 1
                total_words += 1

    print('   %i total words, %i unique words' % (total_words, unique_words))
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    index = len(vocab)
    for word, _ in sorted_word_freqs:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print('  keep the top %i words' % vocab_size)

    # Write (vocab, frequence) to a txt file
    vocab_file = codecs.open('data/%s/vocab' % domain, mode='w', encoding='utf8')
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    for word, index in sorted_vocab:
        if index < 3:
            vocab_file.write(word + '\t' + str(0) + '\n')
            continue
        vocab_file.write(word + '\t' + str(word_freqs[word]) + '\n')
    vocab_file.close()

    return vocab


def read_dataset(domain, phase):
    assert domain in {'restaurant', 'beer'}
    assert phase in {'train', 'test'}

    data_x = []
    data_x_labels = []

    if phase == 'test':
        f_text = codecs.open('data/' + domain + '/' + phase + '.txt', 'r', 'utf-8')
        f_label = codecs.open('data/' + domain + '/' + phase + '_label.txt', 'r', 'utf-8')

        for text, label in zip(f_text, f_label):

            label = label.strip()
            if domain == 'restaurant' and label not in ['Food', 'Staff', 'Ambience']:
                continue

            words = text.strip()

            data_x.append(words)
            data_x_labels.append(label)

        return data_x, data_x_labels

    else:
        assert phase == "train"

        f_text = codecs.open('data/' + domain + '/' + phase + '.txt', 'r', 'utf-8')

        for text in f_text:

            words = text.strip()

            data_x.append(words)

        return data_x


def get_data(domain):
    print('Reading data from', domain)
    # print(' Creating vocab ...')
    # vocab = create_vocab(domain, maxlen, vocab_size)
    print(' Reading dataset ...')
    print('  train set')
    train_x = read_dataset(domain, 'train')
    print('  test set')
    test_x, test_labels = read_dataset(domain, 'test')

    return train_x, test_x, test_labels


if __name__ == "__main__":
    train_data, raw_test_data, test_labels = get_data('beer')

    labels_ids = {}
    for i, l_str in enumerate(sorted(list(set(test_labels)))):
        labels_ids.update({l_str: i})

    raw_test_labels = [labels_ids[l_str] for l_str in test_labels]
    raw_labels = [-1 for _ in train_data]
    raw_data = train_data
    # raw_labels = raw_test_labels

    print("training data size: " + str(len(raw_data)))

    preprocessed_data_set = "JN"
    topic_model_type = "k-components"

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

    print("preprocessed")
    # perform topic modeling based on topic_model_type
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

    elif topic_model_type == "avg_w2v":
        w_d_clustering(data_processed, preprocessed_data_set, vocab, data_processed_labels,
                       test_tokenized_docs, segment_embedding_type="w2v_avg", true_topic_amount=10)

    elif topic_model_type == "doc2vec":
        w_d_clustering(data_processed, preprocessed_data_set, vocab, data_processed_labels,
                       test_tokenized_docs, segment_embedding_type="doc2vec", true_topic_amount=10)


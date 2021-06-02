from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from stop_words import get_stop_words
import nltk
from collections import Counter
from typing import Tuple

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

en_stop = get_stop_words('en')
sw = stopwords.words("english")


stop_words = sw + en_stop
stop_words.append('let')
stop_words.append('gon')
stop_words.append('dhe')
stop_words.extend(['like', 'got',
                   'get', 'one', 'well',
                   'back', 'bit', 'drive',
                   'look', 'see', 'good',
                   'quite', 'think', 'little',
                   'right', 'know', 'thing', 'want'])
stop_words.extend(['put', 'yeah', 'lot''dot', 'le', "'ve", 'really', 'like', 'got', 'get', 'one', 'well',
                   'back', 'bit', 'drive', 'look', 'see', 'good', 'quite', 'think', 'little', 'right', 'know',
                   'thing', 'want', 'dhe', 'gon', 'let', 'get'])
stop_words.extend(["\'re", "n\'t", "n\'t", "'ve", "really", "car", "cars"])


def preprocessing(segments: list, segment_labels: list, preprocessing_type: str, do_stemming: bool = False,
                  remove_low_freq: bool = False, count_threshold: int = 1) \
        -> Tuple[list, list, list, list]:
    """
    preprocessing is used to preprocess the data set
    
    :param segments: raw list of segments
    :param segment_labels: list of segment labels
    :param preprocessing_type: defines the preprocessing approach (["JN", "FP"])
    :param do_stemming: if True, stemming is performed (default: False)
    :param remove_low_freq: if True, all words with absolute frequency under threshold are removed
    :param count_threshold: threshold used when remove_low_freq is set to True

    :return:
        - preprocessed segments
        - labels of preprocessed segments
        - list of vocabulary words
        - list of tokenized 'raw' segments

    """

    if preprocessing_type == "JN":
        do_lemmatizing = False
        do_stop_word_removal = False
    else:
        assert preprocessing_type == "FP"
        do_lemmatizing = True
        do_stop_word_removal = True

    vocabulary = []
    new_docs = []
    new_labels = []
    tokenized_docs = []
    for i, doc in enumerate(segments):

        doc = doc.lower()
        tokens = word_tokenize(doc)

        # remove all tokens that are < 3
        tokens = [w for w in tokens if len(w) > 2]

        # remove all tokens that are just digits
        tokens = [w for w in tokens if w.isalpha()]

        tokenized_doc = [w for w in tokens]

        # remove stop words before stemming/lemmatizing
        if do_stop_word_removal:
            tokens = [w for w in tokens if w not in stop_words]

        # remove all words that are not nouns
        tokens = [w for (w, pos) in nltk.pos_tag(tokens) if pos in ['NN', 'NNP', 'NNS', 'NNPS']]

        # stemming
        if do_stemming:
            tokens = [PorterStemmer().stem(w) for w in tokens]

        # lemmatizing
        if do_lemmatizing:
            tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]

        if len(tokens) == 0:
            continue

        new_docs.append(tokens)
        new_labels.append(segment_labels[i])
        vocabulary.extend(tokens)
        tokenized_docs.append(tokenized_doc)

    if remove_low_freq:
        # remove low-frequency terms

        temp_new_docs = []
        for d in new_docs:
            temp_new_docs.extend(d)
        counter = Counter(temp_new_docs)

        docs_threshold = []
        labels_threshold = []
        vocab_threshold = []
        for i_d, d in enumerate(new_docs):

            d_threshold = [w for w in d if counter[w] > count_threshold]
            if len(d_threshold) > 0:

                labels_threshold.append(new_labels[i_d])
                docs_threshold.append(d_threshold)
                vocab_threshold.extend(d_threshold)

        print("vocab with out threshold len: " + str(len(vocabulary)))
        print("vocab threshold len: " + str(len(vocab_threshold)))
        new_docs = docs_threshold
        vocabulary = vocab_threshold
        new_labels = labels_threshold

    assert len(new_docs) == len(new_labels)
    return new_docs, new_labels, sorted(list(set(vocabulary))), tokenized_docs

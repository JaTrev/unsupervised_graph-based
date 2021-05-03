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


def preprocessing(segments: list, segment_labels: list, do_stemming: bool = False, do_lemmatizing: bool = False,
                  remove_low_freq: bool = False, do_stop_word_removal: bool = True, count_threshold: int = 1) \
        -> Tuple[list, list, list, list]:
    """
    preprocessing is used to preprocess the data set
    
    :param segments: raw list of segments
    :param segment_labels: list of segment labels
    :param do_stemming: if True, stemming is performed (default: False)
    :param do_lemmatizing:  if True, lemmatization is performed (default: False)
    :param remove_low_freq: if True, all words with absolute frequency under threshold are removed
    :param do_stop_word_removal: if True, stop word removal is performed (default: False)
    :param count_threshold: threshold used when remove_low_freq is set to True

    :return:
        - preprocessed segments
        - abels of preprocessed segments
        - list of vocaulary words
        - list of tokenized 'raw' segments

    """

    vocabulary = []
    new_docs = []
    new_labels = []
    tokenized_docs = []
    for i, doc in enumerate(segments):

        doc = doc.lower()

        tkns = word_tokenize(doc)

        # tkns = [w for w in tkns if w not in ['gon', 'na']]

        # remove all tokens that are < 3
        tkns = [w for w in tkns if len(w) > 2]

        # remove all tokens that are just digits
        tkns = [w for w in tkns if w.isalpha()]

        tokenized_doc = [w for w in tkns]

        # remove stop words before stemming/lemmatizing
        if do_stop_word_removal:
            tkns = [w for w in tkns if w not in stop_words]
        else:
            tkns = [w for w in tkns]

        # remove all words that are not nouns
        tkns = [w for (w, pos) in nltk.pos_tag(tkns) if pos in ['NN', 'NNP', 'NNS', 'NNPS']]

        # stemming
        if do_stemming:
            tkns = [PorterStemmer().stem(w) for w in tkns]

        # lemmatizing
        if do_lemmatizing:
            tkns = [WordNetLemmatizer().lemmatize(w) for w in tkns]

        if len(tkns) == 0:
            continue

        new_docs.append(tkns)
        new_labels.append(segment_labels[i])
        vocabulary.extend(tkns)
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

        print("vocab threshold len: " + str(len(vocab_threshold)))
        print("vocab withouth threshold len: " + str(len(vocabulary)))
        new_docs = docs_threshold
        vocabulary = vocab_threshold
        new_labels = labels_threshold

    assert len(new_docs) == len(new_labels)
    return new_docs, new_labels, sorted(list(set(vocabulary))), tokenized_docs

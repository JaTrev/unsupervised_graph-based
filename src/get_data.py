import glob
import os
import pandas as pd
import collections
from tqdm import tqdm
import pickle
from pathlib import Path
from typing import Tuple


def get_partition(task_data_path, path="data/processed_tasks/metadata/partition.csv") \
        -> (dict, collections.defaultdict):
    """
    get_partition fetches information on what sample ID is for training/developing/testing

    :param task_data_path:
    :param path: csv file that maps each sample ID to a train/devel/test
    :return: dicts with mappings between the sample IDs and the proposal
    """

    # any label to collect filenames safely
    names = glob.glob(os.path.join(task_data_path, 'label_segments', 'arousal', '*.' + 'csv'))
    sample_ids = []
    for n in names:
        name_split = n.split(os.path.sep)[-1].split('.')[0]
        sample_ids.append(int(name_split))
    sample_ids = set(sample_ids)

    df = pd.read_csv(path, delimiter=",")
    data = df[["Id", "Proposal"]].values

    id_to_partition = dict()
    partition_to_id = collections.defaultdict(set)

    for i in range(data.shape[0]):
        sample_id = int(data[i, 0])
        partition = data[i, 1]

        if sample_id not in sample_ids:
            continue

        id_to_partition[sample_id] = partition
        partition_to_id[partition].add(sample_id)

    return id_to_partition, partition_to_id


def read_classification_classes(label_file):
    """
    read_classification_classes is used to extract the class_ids from the label file

    :param label_file: path to csv file
    :return: list of class ids
    """

    df = pd.read_csv(label_file, delimiter=",", usecols=['class_id'])
    y_list = df['class_id'].tolist()
    return y_list


def sort_trans_files(elem):
    """
    sort_trans_files is used to calculate a key with which the transcriptions files are sorted

    :param elem: a file name
    :return: file weight used in sorting
    """
    return int(elem.split('_')[-1].split('.')[0])


def prepare_data(task_data_path, transcription_path) -> dict:
    """
    prepare_data creates a dict for the segment-level transcripts and their topic label
    :param task_data_path:
    :param transcription_path:
    :return: dict that consists of transcripts and their topic label
    """
    # Reading transcriptions on SEGMENT-level, sep. in train, develop, test of the official challenge

    id_to_partition, partition_to_id = get_partition(task_data_path)

    data = {}

    # training with test labels available
    for partition in tqdm(partition_to_id.keys()):
        segment_txt = []
        ys_a, ys_v, ys_t = [], [], []

        for sample_id in tqdm(sorted(partition_to_id[partition])):
            transcription_files = glob.glob(os.path.join(transcription_path, str(sample_id), '*.' + 'csv'))

            for file in sorted(transcription_files, key=sort_trans_files):
                df = pd.read_csv(file, delimiter=',')
                words = df['word'].tolist()
                segment_txt.append(" ".join(words))

            # training without test labels available
            label_file_topic = os.path.join(task_data_path, 'label_segments', 'topic', str(sample_id) + ".csv")
            y_list_topic = read_classification_classes(label_file_topic)

            for y in y_list_topic:
                ys_t.append(y)

        data[partition] = {'text': segment_txt, 'labels_topic': ys_t}

    return data


def get_data(data_set: str, get_test_data: bool, task_data_path='data/processed_tasks/c2_muse_topic',
             transcription_path='data/processed_tasks/c2_muse_topic/transcription_segments') \
        -> Tuple[list, list, list, list]:
    """
    get_data collects the data and test_data

    :param data_set: name of the data set (MUSE or CRR)
    :param get_test_data: getting test data flag
    :param task_data_path:  path to data task
    :param transcription_path: path to transcription

    :return:
        - training data -
        - training labels -
        - testing data -
        - testing labels -
    """

    if data_set == "CRR":
        # using the Citysearch Restaurant Reviews corpus
        data, test_data = ([], [])
        with open("data/restaurant/test.txt") as f:
            for line in f:
                if "\n" != line:
                    test_data.append(line.replace("\n", ""))

        with open("data/restaurant/train.txt") as f:
            for line in f:
                if "\n" != line:
                    data.append(line.replace("\n", ""))

        test_labels = [-1 for _ in range(len(test_data))]
        labels = [-1 for _ in range(len(data))]

        return data, labels, test_data, test_labels

    if Path("data/saved_data.pickle").is_file():

        with open("data/saved_data.pickle", "rb") as myFile:
            data = pickle.load(myFile)

        with open("data/saved_data_labels.pickle", "rb") as myFile:
            data_labels = pickle.load(myFile)

        if get_test_data:

            with open("data/saved_test_data.pickle", "rb") as myFile:
                test_data = pickle.load(myFile)

            with open("data/saved_test_labels.pickle", "rb") as myFile:
                test_data_label = pickle.load(myFile)
        else:
            test_data = None
            test_data_label = None

    else:

        all_data = prepare_data(task_data_path=task_data_path, transcription_path=transcription_path)

        data = all_data['train']['text']
        data.extend(all_data['devel']['text'])

        data_labels = all_data['train']['labels_topic']
        data_labels.extend(all_data['devel']['labels_topic'])

        with open("data/saved_data.pickle", "wb") as myFile:
            pickle.dump(data, myFile)

        with open("data/saved_data_labels.pickle", "wb") as myFile:
            pickle.dump(data_labels, myFile)

        if get_test_data:
            test_data = all_data['test']['text']
            test_data_label = all_data['test']['labels_topic']

            with open("data/saved_test_data.pickle", "wb") as myFile:
                pickle.dump(test_data, myFile)

            with open("data/saved_test_labels.pickle", "wb") as myFile:
                pickle.dump(test_data_label, myFile)
        else:
            test_data = None
            test_data_label = None

    filtered_data, filtered_data_labels = filter_dataset(data, data_labels)

    if get_test_data:
        filtered_test_data, filtered_test_data_labels = filter_dataset(test_data, test_data_label)
    else:
        filtered_test_data, filtered_test_data_labels = (None, None)

    return filtered_data, filtered_data_labels, filtered_test_data, filtered_test_data_labels


def filter_dataset(data: list, data_labels: list) -> (list, list):
    """
    filter_dataset returns a set of segments without the very short segments,
    and their respective labels

    :param data: segment data set
    :param data_labels: set of data labels
    :return:
        - filtered_data - segment data set without the short segments
        - filtered_data_labels -  labels of the returning segment data set
    """
    new_data, new_data_labels = zip(*((segment, label) for segment, label
                                      in zip(data, data_labels)
                                      if len([w for w in segment.split() if w.isalpha()]) > 2))

    return new_data, new_data_labels

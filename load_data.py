import os

import numpy as np
import pandas as pd
from numpy import random

DATA_DIR = "{}{}{}{}".format(os.getcwd(), os.sep, 'data', os.sep)


def split_to_sentences(data):
    """

    :param data:
    :return:
    """
    sentences = map(lambda s: s.strip(), data.split('.'))

    return list(sentences)


def split_in_train_dev_test(sentences, seed=1234, dev_size=0.20, test_size=.10, save_data=False):
    """

    :param sentences:
    :param seed:
    :param dev_size:
    :param test_size:
    :param save_data:
    :return:
    """
    # setting the seed in order to be able to reproduce results.
    np.random.seed(seed)

    # shuffling the list of sentences.
    random.shuffle(sentences)

    total_len = len(sentences)
    dev_sentences_size = int(total_len * dev_size)
    test_sentences_size = int(total_len * test_size)
    train_sentences_size = total_len - dev_sentences_size - test_sentences_size

    train_data = sentences[:train_sentences_size]
    dev_data = sentences[train_sentences_size:train_sentences_size + dev_sentences_size]
    test_data = sentences[train_sentences_size + dev_sentences_size:]

    if save_data:
        train_df = pd.DataFrame(train_data, columns=['text'])
        dev_df = pd.DataFrame(dev_data, columns=['text'])
        test_df = pd.DataFrame(test_data, columns=['text'])

        train_df.to_csv(DATA_DIR + 'europarl_train.csv', encoding='utf-8', index=False)
        dev_df.to_csv(DATA_DIR + 'europarl_dev.csv', encoding='utf-8', index=False)
        test_df.to_csv(DATA_DIR + 'europarl_test.csv', encoding='utf-8', index=False)

    return train_data, dev_data, test_data


def load_dataset():
    """
    This method loads the english dataset.
    :return:
    """
    infile = DATA_DIR + 'europarl-v7.el-en.en'

    with open(infile, 'rb') as in_file:
        dataset = in_file.read().decode("utf-8")

    return dataset


if __name__ == "__main__":

    en_data = load_dataset()

    en_sentences = split_to_sentences(en_data)
    train, dev, test = split_in_train_dev_test(en_sentences, save_data=True)

    for sentence in train[:10]:
        print(sentence.strip(), end='\n\n')

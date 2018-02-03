from Assignment1 import DATA_DIR, setup_logger

import numpy as np
import pandas as pd
from numpy import random
from pprint import pprint

logger = setup_logger(__name__)
logger.disabled = True


class Fetcher(object):

    def __init__(self, file, language):
        """

        :param file:
        :param language:
        """
        self.file = file
        self.language = language
        self.train_data = None
        self.test_data = None
        self.dev_data = None

    def load_dataset(self):
        """
        This method loads the english dataset.
        :return:
        """
        infile = "{0}{1}{2}".format(DATA_DIR, self.file, self.language)
        logger.info('Loading File: {}'.format(infile))

        with open(infile, 'rb') as in_file:
            dataset = in_file.read().decode("utf-8")

        return dataset

    def split_in_train_dev_test(self,
                                sentences,
                                seed=1234,
                                dev_size=0.20,
                                test_size=.10,
                                save_data=False):
        """
        This method, given a seed, splits an iterable of sentences into training, development and test sets.

        :param sentences: List. An iterable of sentences (strings)
        :param seed: Int. A number that helps in the reproduction of the samples
        :param dev_size: Float. The size of the development dataset. Must be in (0, 1)
        :param test_size: Float. The size of the test dataset. Must be in (0, 1)
        :param save_data: Bool. Whether we want to save the constructed datasets in .csv form.
        :return:
        """

        logger.info('Seed for reproducibility of the split: {}'.format(seed))
        # setting the seed in order to be able to reproduce results.
        np.random.seed(seed)

        # shuffling the list of sentences.
        random.shuffle(sentences)

        # calculating the ratios in actual numbers
        total_len = len(sentences)
        dev_sentences_size = int(total_len * dev_size)
        test_sentences_size = int(total_len * test_size)
        train_sentences_size = total_len - dev_sentences_size - test_sentences_size

        logger.info('Total number of sentences: {}'.format(len))
        logger.info('Training Dataset Size: {}'.format(train_sentences_size))
        logger.info('Development Dataset Size: {}'.format(dev_sentences_size))
        logger.info('Test Dataset Size: {}'.format(test_sentences_size))

        # splitting the data to train, development and test
        train_data = sentences[:train_sentences_size]
        dev_data = sentences[train_sentences_size:train_sentences_size + dev_sentences_size]
        test_data = sentences[train_sentences_size + dev_sentences_size:]

        assert len(train_data) == train_sentences_size
        assert len(dev_data) == dev_sentences_size
        assert len(test_data) == test_sentences_size

        if save_data:
            train_df = pd.DataFrame(train_data, columns=['text'])
            dev_df = pd.DataFrame(dev_data, columns=['text'])
            test_df = pd.DataFrame(test_data, columns=['text'])

            logger.info('Saving datasets as .csv files.')

            train_df.to_csv(DATA_DIR + 'europarl_train.csv', encoding='utf-8', index=False)
            dev_df.to_csv(DATA_DIR + 'europarl_dev.csv', encoding='utf-8', index=False)
            test_df.to_csv(DATA_DIR + 'europarl_test.csv', encoding='utf-8', index=False)

        self.train_data = train_data
        self.test_data = test_data
        self.dev_data = dev_data

    def split_in_train_test(self,
                            sentences,
                            seed=1234,
                            test_size=.10,
                            save_data=False):
        """

        :param sentences:
        :param seed:
        :param test_size:
        :param save_data:
        :return:
        """
        logger.info('Seed for reproducibility of the split: {}'.format(seed))
        # setting the seed in order to be able to reproduce results.
        np.random.seed(seed)

        # shuffling the list of sentences.
        random.shuffle(sentences)

        # calculating the ratios in actual numbers
        total_len = len(sentences)
        test_sentences_size = int(total_len * test_size)
        train_sentences_size = total_len - test_sentences_size

        logger.info('Total number of sentences: {}'.format(len))
        logger.info('Training Dataset Size: {}'.format(train_sentences_size))
        logger.info('Test Dataset Size: {}'.format(test_sentences_size))

        # splitting the data to train, development and test
        train_data = sentences[:train_sentences_size]
        test_data = sentences[train_sentences_size:]

        assert len(train_data) == train_sentences_size
        assert len(test_data) == test_sentences_size

        if save_data:
            train_df = pd.DataFrame(train_data, columns=['text'])
            test_df = pd.DataFrame(test_data, columns=['text'])

            logger.info('Saving Data-sets as .csv files.')

            train_df.to_csv(DATA_DIR + 'europarl_train.csv', encoding='utf-8', index=False)
            test_df.to_csv(DATA_DIR + 'europarl_test.csv', encoding='utf-8', index=False)

        self.train_data = train_data
        self.test_data = test_data

    @staticmethod
    def feed_cross_validation(sentences,
                              seed=1234,
                              k_folds=5):

        """
        This method feeds the train and held_out data-sets in each iteration.
        :param sentences: list. An iterable of sentences
        :param seed: Int. A number that helps in the reproduction of the shuffling
        :param k_folds: Int. Number of folds.
        :return: yields a dict of train and held_out data-sets.
        """

        # setting the seed in order to be able to reproduce results.
        np.random.seed(seed)

        # shuffling the list of sentences.
        logger.info('Shuffling data set in order to brake to train and held_out data-sets')
        random.shuffle(sentences)

        # calculating the ratios in actual numbers
        total_len = len(sentences)

        # split_size
        split_size = int(total_len / float(k_folds))
        logger.info('Spliting data-set in {} folds'.format(k_folds))
        logger.info('Split size for held out dataset: {}'.format(split_size))

        for i in range(1, k_folds + 1):

            yield {'held_out': sentences[(i - 1) * split_size: i * split_size],
                   'train': sentences[:(i - 1) * split_size] + sentences[i * split_size:]}


if __name__ == "__main__":

    dl = Fetcher(file='europarl-v7.el-en.',
                 language='en')

    test_sentences = ["the sentence number {}".format(i) for i in range(1, 22)]
    for i in dl.feed_cross_validation(sentences=test_sentences):
        pprint(i)

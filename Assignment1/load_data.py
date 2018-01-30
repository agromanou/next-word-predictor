from Assignment1 import DATA_DIR
from Assignment1.text_utils import TextUtils

import numpy as np
import pandas as pd
from numpy import random


class DataFetcher:

    def __init__(self, file, language):
        self.file = file
        self.language = language
        self.train_data = None
        self.test_data = None
        self.dev_data = None
        self.vocabulary = None
        self.tokens = None

    def load_dataset(self):
        """
        This method loads the english dataset.
        :return:
        """
        infile = DATA_DIR + self.file + self.language

        with open(infile, 'rb') as in_file:
            dataset = in_file.read().decode("utf-8")

        return dataset

    def split_in_train_dev_test(self, sentences,
                                seed=1234,
                                dev_size=0.20,
                                test_size=.10,
                                save_data=False):
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

        # calculating the ratios in actual numbers
        total_len = len(sentences)
        dev_sentences_size = int(total_len * dev_size)
        test_sentences_size = int(total_len * test_size)
        train_sentences_size = total_len - dev_sentences_size - test_sentences_size

        # splitting the data to train, development and test
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

        self.train_data = train_data
        self.test_data = test_data
        self.dev_data = test_data


if __name__ == "__main__":

    utils = TextUtils()
    dl = DataFetcher('europarl-v7.el-en.', 'en')

    # Load data & split sentences
    en_data = dl.load_dataset()
    en_sentences = utils.split_to_sentences(en_data)

    # Split data into train, dev and test
    dl.split_in_train_dev_test(en_sentences, save_data=True)
    train, dev, test = dl.train_data, dl.dev_data, dl.test_data

    for sentence in train[:10]:
        print(sentence.strip(), end='\n\n')

    a_sentence = "This is a quite large sentence"

    some_sentences = ["This is a sentence",
                      "This is another sentence",
                      "This is new fucking awesome sentence"]

    # Tokenize sentences
    res = utils.tokenize_and_pad(a_sentence)

    # Create vocabulary
    counts = utils.create_vocabulary(some_sentences)

    # Create n-grams
    for i in utils.create_ngrams(a_sentence.split(), 3):
        print(i)

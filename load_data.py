import os
from collections import Counter
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


def tokenize_and_pad(sentence, model_type='simple'):
    """

    :param sentence:
    :param model_type:
    :return:
    """
    assert model_type in ['bigram', 'trigram', 'simple']

    words = sentence.split()

    if model_type == 'bigram':
        return ['Start1'] + words + ['End1']

    elif model_type == 'trigram':
        return ['Start1', 'Start2'] + words + ['End1', 'End2']

    return words


def split_in_train_dev_test(sentences,
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


def create_vocabulary(sentences, base_limit=2):
    """
    This method counts all the tokens from a list of sentences. Then it creates a vocabulary with the most common
    tokens, that surpass the base limit.
    :param sentences: list. A list of strings.
    :param base_limit: int. A number defining the base limit for the validity of the tokens.
    :return: dict. A dictionary with the vocabulary and the rejected tokens
    """
    # grab all the tokens in an iterator. Not in a list.
    tokens = (token for sentence in sentences for token in tokenize_and_pad(sentence, model_type='simple'))

    tokens_count = Counter(tokens)

    valid_tokens = {k: v for k, v in tokens_count.items() if v > base_limit}
    invalid_tokens = {k: v for k, v in tokens_count.items() if v <= base_limit}

    return dict(vocabulary=valid_tokens,
                rejected=invalid_tokens)


if __name__ == "__main__":
    # en_data = load_dataset()
    #
    # en_sentences = split_to_sentences(en_data)
    # train, dev, test = split_in_train_dev_test(en_sentences, save_data=True)
    #
    # for sentence in train[:10]:
    #     print(sentence.strip(), end='\n\n')

    a_sentence = "This is a sentence"

    some_sentences = ["This is a sentence",
                      "This is another sentence",
                      "This is new fucking awesome sentence"]

    res = tokenize_and_pad(a_sentence)
    print(res)

    counts = create_vocabulary(some_sentences)
    print(counts)
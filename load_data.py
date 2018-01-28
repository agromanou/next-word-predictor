import os
import nltk

DATA_DIR = "{}{}{}{}".format(os.getcwd(), os.sep, 'data', os.sep)


def load_dataset():
    """
    This method loads the english dataset.
    :return:
    """
    infile = DATA_DIR + 'europarl-v7.el-en.en'

    with open(infile, 'rb') as in_file:
        dataset = in_file.read()

    return dataset


if __name__ == "__main__":

    data = load_dataset()

    print(data)

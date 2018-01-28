import os
import nltk

DATA_DIR = "{}{}{}{}".format(os.getcwd(), os.sep, 'data', os.sep)

infile = DATA_DIR + 'europarl-v7.el-en.en'

with open(infile, 'rb') as in_file:
    data = in_file.read()
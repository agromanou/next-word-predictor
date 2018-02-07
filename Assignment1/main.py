from Assignment1.data_fetcher import Fetcher
from Assignment1.preprocess import Preprocessor

if __name__ == '__main__':

    # pp = Preprocessor()
    #
    # dl = Fetcher(file='europarl-v7.el-en.',
    #              language='en')
    #
    # # Load data & split sentences
    # en_data = dl.load_dataset()
    # en_sentences = pp.split_to_sentences(en_data)
    # print(len(en_sentences))
    #
    # # Split data into train, dev and test
    # dl.split_in_train_dev_test(en_sentences,
    #                            save_data=False)

    # train, dev, test = dl.train_data, dl.dev_data, dl.test_data

    # # Tokenize sentences
    # res = pp.tokenize_and_pad(a_sentence)
    #
    # # Create vocabulary
    # counts = pp.create_vocabulary(some_sentences)
    #
    # # Create n-grams
    # for i in pp.create_ngrams(a_sentence.split(), 3):
    #     print(i)

    a_sentence = "This is a huge sentence that I want to exclude many different words for some reason"

    import re

    out = a_sentence.replace('reason|wer', 'AAAA')

    print(out)
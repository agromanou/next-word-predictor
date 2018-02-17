from Assignment1.preprocess import Preprocessor
from Assignment1.data_fetcher import Fetcher
from pprint import pprint
import re
import itertools

if __name__ == '__main__':

    mod_type = 'trigram'

    dl_obj = Fetcher(file='europarl-v7.el-en.', language='en')

    pre_obj = Preprocessor()

    # loading the whole dataset.
    dataset = dl_obj.load_dataset()

    # Splitting the whole dataset into sentences, in order to then split into train and test.
    sentences = pre_obj.split_to_sentences(dataset)[:1000]

    padded_sentences = [pre_obj.tokenize_and_pad(sentence=s, model_type=mod_type) for s in sentences]

    # Splitting in train and test sentences. Everything will be stored in Fetcher's object.
    dl_obj.split_in_train_test(padded_sentences)


    train_sentences = dl_obj.train_data

    train_tokens = itertools.chain(*train_sentences)

    pre_obj.calculate_ngram_counts(mod_type,
                                   padded_sentences,
                                   base_limit=3)
    # print(len(vocabulary_tokens_counts))
    # print(len(rejection_tokens_counts))

    for i in dl_obj.feed_cross_validation(sentences=train_sentences):

        train = i["train"]
        dev = i['held_out']
        # print(train)

        # count tokens, and create vocabulary-rejection lexicons

        # # replacing uncommon words in original corpus:
        # new_corpus = pre_obj.replace_uncommon_words(corpus=train_corpus,
        #                                             words=rejection_tokens_counts.keys(),
        #                                             replacement='<UNK>')
        # pprint(new_corpus)
        break

    # pp = Preprocessor()
    #
    # # Load data & split sentences
    # en_data = dl.load_dataset()
    # en_sentences = pp.split_to_sentences(en_data)
    #
    # # Split data into train, dev and test
    # dl.split_in_train_dev_test(en_sentences,
    #                            save_data=True)
    #
    # train, dev, test = dl.train_data, dl.dev_data, dl.test_data
    #
    # for sentence in train[:10]:
    #     print(sentence.strip(), end='\n\n')
    #
    # a_sentence = "This is a quite large sentence"
    #
    # some_sentences = ["This is a sentence",
    #                   "This is another sentence",
    #                   "This is new fucking awesome sentence"]
    #
    # # Tokenize sentences
    # res = pp.tokenize_and_pad(a_sentence)
    #
    # # Create vocabulary
    # counts = pp.create_vocabulary(some_sentences)
    #
    # # Create n-grams
    # for i in pp.create_ngrams(a_sentence.split(), 3):
    #     print(i)

from Assignment1.preprocess import Preprocessor
from Assignment1.data_fetcher import Fetcher
from Assignment1.modelling import Model
from Assignment1 import setup_logger

logger = setup_logger(__name__)
logger.disabled = True

if __name__ == '__main__':
    """
    1) Corpus to tokens.
    2) Count token frequencies and create vocabulary and discarded (implemented)
    3) Replace in corpus the discarded tokens with "unk"
    4) Split in sentences
    5) Split each sentence in tokens
    6) Create n-grams
    """

    dl = Fetcher(file='europarl-v7.el-en.',
                 language='en')
    pp = Preprocessor()

    # Load data & split sentences
    en_data = dl.load_dataset()
    en_sentences = pp.split_to_sentences(en_data)

    # Split data into train, dev and test
    dl.split_in_train_dev_test(en_sentences,
                               save_data=True)

    train, dev, test = dl.train_data, dl.dev_data, dl.test_data

    # Process with unigrams
    corpus_train = pp.flatten_to_one_corpus(train[:3])
    print(corpus_train)
    tokens = pp.tokenize_and_pad(corpus_train)
    print(tokens)
    voc = pp.create_vocabulary(tokens, 0)
    print(voc["vocabulary"])

    model = Model("laplace_smoothing", voc["vocabulary"], tokens)
    model.perform_smoothing()
    print(model.prob)



    # for sentence in train[:10]:
    #     res.append(sentence.strip())

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

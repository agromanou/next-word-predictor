from Assignment1.preprocess import Preprocessor
from Assignment1.data_fetcher import Fetcher

if __name__ == '__main__':

    dl_obj = Fetcher(file='europarl-v7.el-en.', language='en')

    pre_obj = Preprocessor()

    dataset = dl_obj.load_dataset()
    sentences = pre_obj.split_to_sentences(dataset)
    dl_obj.split_in_train_test(sentences)

    train_sentences = dl_obj.train_data[:10]

    for i in dl_obj.feed_cross_validation(sentences=train_sentences):
        train = i["train"]
        dev = i['held_out']
        break

    train_corpus = pre_obj.flatten_to_one_corpus(train)
    tokens = pre_obj.tokenize_and_pad(train_corpus)

    



    print(train_corpus)


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

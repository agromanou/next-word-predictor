from Assignment1.preprocess import Preprocessor
from Assignment1.data_fetcher import Fetcher

if __name__ == '__main__':
    pp = Preprocessor()
    dl = Fetcher('europarl-v7.el-en.', 'en')

    # Load data & split sentences
    en_data = dl.load_dataset()
    en_sentences = pp.split_to_sentences(en_data)

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
    res = pp.tokenize_and_pad(a_sentence)

    # Create vocabulary
    counts = pp.create_vocabulary(some_sentences)

    # Create n-grams
    for i in pp.create_ngrams(a_sentence.split(), 3):
        print(i)
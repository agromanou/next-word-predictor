from itertools import chain

import numpy as np

from Assignment1 import setup_logger
from Assignment1.app import prepare_test_metadata
from Assignment1.data_fetcher import Fetcher
from Assignment1.evaluation import Evaluation
from Assignment1.modelling import Model
from Assignment1.preprocess import Preprocessor

logger = setup_logger(__name__)


def run_final_model(mod_type='bigram',
                    smoothing='laplace_smoothing',
                    threshold=3,
                    n_sentences=1000,
                    n_random_sentences_check=5):
    """

    :param mod_type:
    :param smoothing:
    :param threshold:
    :param n_sentences:
    :param n_random_sentences_check:
    :return:
    """

    logger.info('Running Model for given parameters: ')
    logger.info('Model N-Grams: {0}, Smoothing Type: {1}, Vocabulary Threshold: {2}, Number of Sentences: {3}.'.format(
        mod_type.title(), smoothing.title(), threshold, n_sentences))

    dl_obj = Fetcher(file='europarl-v7.el-en.', language='en')

    model_n = 2 if mod_type == 'bigram' else 3

    pre_obj = Preprocessor()

    # loading the whole dataset.
    dataset = dl_obj.load_dataset()
    # Splitting the whole dataset into sentences, in order to then split into train and test.
    sentences = pre_obj.split_to_sentences(dataset)

    logger.info('Extracting {} sentences for execution.'.format(n_sentences))
    sentences = sentences[:n_sentences]

    logger.info('Applying tokenization and padding for {} sentences'.format(len(sentences)))
    padded_sentences = [pre_obj.tokenize_and_pad(sentence=s, model_type=mod_type) for s in sentences]

    # Splitting in train and test sentences. Everything will be stored in Fetcher's object.
    dl_obj.split_in_train_test(padded_sentences)

    training_ngram_counts, rejected_tokens = pre_obj.create_ngram_metadata(mod_type,
                                                                           dl_obj.train_data,
                                                                           threshold=threshold)

    test_prepared_ngrams, test_unigrams_counter = prepare_test_metadata(
        iterable_of_sentence_tokens=dl_obj.test_data,
        rejected_words=rejected_tokens,
        model_n=model_n)

    model_obj = Model(model_ngrams=training_ngram_counts, n_model=model_n)

    model_obj.fit_model(smoothing_algo=smoothing)

    test_probs = model_obj.get_test_ngrams_smoothed_probs(test_ngram_tuples=test_prepared_ngrams)

    eval_obj = Evaluation(model_to_test=test_probs,
                          data_to_test=test_prepared_ngrams,
                          tokens_count=test_unigrams_counter)

    eval_obj.compute_model_performance()

    logger.info('Avg Cross Entropy: {}'.format(eval_obj.cross_entropy))
    logger.info('Avg Perplexity: {}'.format(eval_obj.perplexity))

    vocabulary_words = set(chain.from_iterable(model_obj.vocabulary.keys())) - {'<s1>', '<s2>', '</s1>', '</s1>'}

    np.random.seed(1234)
    random_sentences = np.random.choice(a=dl_obj.test_data,
                                        size=n_random_sentences_check,
                                        replace=False)

    for sentence_tokens in random_sentences:
        model_obj.evaluate_on_sentences_pairs(sentence=sentence_tokens,
                                              vocabulary=list(vocabulary_words))

    return {
        'model_obj': model_obj,
        'eval_obj': eval_obj,
        'cross_entropy': eval_obj.cross_entropy,
        'perplexity': eval_obj.perplexity
    }


if __name__ == '__main__':
    mod_type = 'bigram'
    smoothing = 'laplace_smoothing'
    baselim = 10
    nsentences = 10000

    print("The model is trained. Please wait for you input...")
    obj = run_final_model(mod_type=mod_type,
                          smoothing=smoothing,
                          threshold=baselim,
                          n_sentences=nsentences)

    # # Run keyword predictor
    # print("Model have been trained!")
    # word = input("Please type a word \n(To exit type the word: <exit>) \n\n")
    # while word != '<exit>':
    #     mle_dict = obj['model_obj'].mle_predict_word((word,))
    #     print(mle_dict)
    #     word = input("Please type a word \n(To exit type the word: <exit>)\n\n ")
    #

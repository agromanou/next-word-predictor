import numpy as np
from Assignment1.data_fetcher import Fetcher
from Assignment1.evaluation import Evaluation
from Assignment1.modelling import Model
from Assignment1.preprocess import Preprocessor
from Assignment1 import setup_logger
from pprint import pprint
from Assignment1.app import prepare_test_metadata

logger = setup_logger(__name__)


def run_final_model(mod_type='bigram',
                    smoothing='laplace_smoothing',
                    baselim=5,
                    n_sentences=10000,
                    n_folds=5):
    """

    :param mod_type:
    :param smoothing:
    :param baselim:
    :param n_sentences:
    :param n_folds:
    :return:
    """

    logger.info('Running Model for given parameters: ')
    logger.info('Model N-Grams: {0}, Smoothing Type: {1}, Vocabulary Base Limit: {2}, Number of Sentences: {3}.'.format(
        mod_type.title(), smoothing.title(), baselim, n_sentences))

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
                                                                           base_limit=baselim)

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

    logger.info('Avg Cross Entropy: ', eval_obj.cross_entropy)
    logger.info('Avg Perplexity: ', eval_obj.perplexity)

    return {
        'model_obj': model_obj,
        'eval_obj': eval_obj,
        'cross_entropy': eval_obj.cross_entropy,
        'perplexity': eval_obj.perplexity
    }


if __name__ == '__main__':
    mod_type = 'trigram'
    smoothing = 'laplace_smoothing'
    baselim = 10
    nsentences = 10000

    run_final_model(mod_type=mod_type,
                    smoothing=smoothing,
                    baselim=baselim,
                    n_sentences=nsentences,
                    n_folds=5)

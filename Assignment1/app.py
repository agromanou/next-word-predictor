import itertools

import numpy as np

from Assignment1.data_fetcher import Fetcher
from Assignment1.evaluation import Evaluation
from Assignment1.modelling import Model
from Assignment1.preprocess import Preprocessor

from pprint import pprint


def prepare_test_metadata(iterable_of_sentence_tokens, rejected_words, model_n, replacement='<UNK>'):
    """

    :param iterable_of_sentence_tokens:
    :param rejected_words:
    :param model_n:
    :param replacement:
    :return:
    """
    out = list()
    counter = 0
    for sentence_list in iterable_of_sentence_tokens:

        for num in range(len(sentence_list)):
            if sentence_list[num] in rejected_words:
                sentence_list[num] = replacement

            counter += 1

        ngrams = Preprocessor.create_ngrams(seq=sentence_list, n=model_n)
        out.extend(ngrams)

    return out, counter


def run_example(mod_type='bigram', smoothing='laplace_smoothing', baselim=5, n_sentences=10000, n_folds=5):
    """

    :param mod_type:
    :param smoothing:
    :param baselim:
    :param n_sentences:
    :param n_folds:
    :return:
    """

    dl_obj = Fetcher(file='europarl-v7.el-en.', language='en')

    model_n = 2 if mod_type == 'bigram' else 3

    pre_obj = Preprocessor()

    # loading the whole dataset.
    dataset = dl_obj.load_dataset()

    # Splitting the whole dataset into sentences, in order to then split into train and test.
    sentences = pre_obj.split_to_sentences(dataset)[:n_sentences]

    padded_sentences = [pre_obj.tokenize_and_pad(sentence=s, model_type=mod_type) for s in sentences]

    # Splitting in train and test sentences. Everything will be stored in Fetcher's object.
    dl_obj.split_in_train_test(padded_sentences)

    train_sentences = dl_obj.train_data

    cross_entropy_res, perplexity_res = list(), list()

    for fold in dl_obj.feed_cross_validation(sentences=train_sentences,
                                             seed=1234,
                                             k_folds=n_folds):
        train_padded_sentences = fold["train"]
        dev_padded_sentences = fold['held_out']

        training_ngram_counts, rejected_tokens = pre_obj.create_ngram_metadata(mod_type,
                                                                               train_padded_sentences,
                                                                               base_limit=baselim)

        dev_prepared_ngrams, test_unigrams_counter = prepare_test_metadata(
            iterable_of_sentence_tokens=dev_padded_sentences,
            rejected_words=rejected_tokens,
            model_n=model_n)

        model_obj = Model(ngrams=training_ngram_counts)

        model_obj.fit_model(smoothing_algo=smoothing)

        probs = model_obj.probs

        eval_obj = Evaluation(probs, dev_prepared_ngrams, test_unigrams_counter)

        eval_obj.compute_model_performance()
        cross_entropy_res.append(eval_obj.cross_entropy)
        perplexity_res.append(eval_obj.perplexity)

    mean_cross_entropy = np.mean(cross_entropy_res)
    mean_perplexity = np.mean(perplexity_res)

    print('Avg Cross Entropy: ', mean_cross_entropy)
    print('Avg Perplexity: ', mean_perplexity)


if __name__ == '__main__':
    mod_type = 'bigram'
    smoothing = 'laplace_smoothing'
    baselim = 10
    nsentences = 100000

    run_example(mod_type=mod_type,
                smoothing=smoothing,
                baselim=baselim,
                n_sentences=nsentences,
                n_folds=5)

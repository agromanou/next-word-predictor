import numpy as np

from Assignment1 import setup_logger
from Assignment1.data_fetcher import Fetcher
from Assignment1.evaluation import Evaluation
from Assignment1.modelling import Model
from Assignment1.preprocess import Preprocessor

logger = setup_logger(__name__)


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

    logger.info('Number of test tokens: {}'.format(counter))

    return out, counter


def run_example(mod_type='bigram',
                smoothing='laplace_smoothing',
                threshold=5,
                n_sentences=10000,
                n_folds=5,
                interpolation=False):
    """

    :param mod_type:
    :param smoothing:
    :param threshold:
    :param n_sentences:
    :param n_folds:
    :param interpolation:
    :return:
    """

    logger.info('Running Model for given parameters: ')
    logger.info('Model N-Grams: {0},'
                ' Smoothing Type: {1},'
                ' Vocabulary Base Limit: {2},'
                ' Number of Sentences: {3},'
                ' Interpolation: {}.'.format(
        mod_type.title(), smoothing.title(), threshold, n_sentences, interpolation))

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

    # placeholders fro k fold cross entropy and perplexity
    cross_entropy_res, perplexity_res = list(), list()

    for fold in dl_obj.feed_cross_validation(sentences=dl_obj.train_data,
                                             seed=1234,
                                             k_folds=n_folds):

        # getting the two data-sets, train and held-out.
        train_padded_sentences = fold["train"]
        dev_padded_sentences = fold['held_out']

        # counting ngrams, and finding rejected tokens
        training_ngram_counts, rejected_tokens = pre_obj.create_ngram_metadata(mod_type,
                                                                               train_padded_sentences,
                                                                               threshold=threshold)

        # creating ngrams, and unigram counts for dev (held out) set
        dev_prepared_ngrams, dev_unigrams_counter = prepare_test_metadata(
            iterable_of_sentence_tokens=dev_padded_sentences,
            rejected_words=rejected_tokens,
            model_n=model_n)

        # instantiating the Model class in order to train our model and calculate the probabilities.
        model_obj = Model(model_ngrams=training_ngram_counts,
                          n_model=model_n,
                          interpolation=interpolation)

        # fitting the Model
        model_obj.fit_model(smoothing_algo=smoothing)

        # Obtain the dev (held out) probabilities
        dev_probs = model_obj.get_test_ngrams_smoothed_probs(test_ngram_tuples=dev_prepared_ngrams)

        # Instantiating the Evaluation class in order to test our model.
        eval_obj = Evaluation(model_to_test=dev_probs,
                              data_to_test=dev_prepared_ngrams,
                              tokens_count=dev_unigrams_counter)

        # computing the overall model performance.
        eval_obj.compute_model_performance()

        # appending the k-fold cross entropy and perplexity.
        cross_entropy_res.append(eval_obj.cross_entropy)
        perplexity_res.append(eval_obj.perplexity)

    # calculating average cross entropy and perplexity.
    mean_cross_entropy = np.mean(cross_entropy_res)
    mean_perplexity = np.mean(perplexity_res)

    logger.info('Avg Cross Entropy: {}'.format(mean_cross_entropy))
    logger.info('Avg Perplexity: {}'.format(mean_perplexity))


if __name__ == '__main__':

    mod_type = 'trigram'
    smoothing = 'laplace_smoothing'
    baselim = 10
    nsentences = 10000

    run_example(mod_type=mod_type,
                smoothing=smoothing,
                threshold=baselim,
                n_sentences=nsentences,
                n_folds=5)

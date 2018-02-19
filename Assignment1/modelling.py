from Assignment1 import setup_logger
from Assignment1.preprocess import Preprocessor
from Assignment1.evaluation import Evaluation

from collections import defaultdict
from pprint import pprint
import numpy as np
import operator
import random

logger = setup_logger(__name__)


class Model(object):
    def __init__(self,
                 model_ngrams,
                 n_model=2,
                 laplace_k=2,
                 kneser_ney_d=0.75,
                 interpolation=False):

        self.n_model = n_model
        self.interpolation = interpolation
        self.laplace_k = laplace_k
        self.knener_ney_d = kneser_ney_d

        self.model_ngrams_raw = model_ngrams
        self.vocabulary = model_ngrams[1]  # will always be the 1-gram counts.
        self.model_ngrams = self.merge_ngram_counts(model_ngrams)  # merged ngram tuples.

        self.tokens_count = sum(self.vocabulary.values())

        self.probs = dict()
        self.test_probs = dict()
        self.smoothed_probs = dict()
        self.interpolated_probs = dict()
        self.trained_model = dict()

    @staticmethod
    def merge_ngram_counts(ngrams):
        """
        Takes as input a nested dictionary and flattens it.
        :return: A flattened dictionary
        """
        super_dict = defaultdict(int)  # uses set to avoid duplicates

        list_of_dicts = ngrams.values()

        for d in list_of_dicts:
            for k, v in d.items():
                super_dict[k] = v

        return super_dict

    def fit_model(self, smoothing_algo):
        """
        This method runs the modeling process by calculating the Bayes probabilities
        and performing smoothing on the Models variables.
        :param smoothing_algo: The name of the smoothing algorithm that will be used.
        """
        assert smoothing_algo in ['laplace_smoothing', 'k_n']

        # calculation bayes probabilities
        self.probs = self.calculate_bayes_probs()

        # performing smoothing
        self.perform_smoothing(smoothing_algo)

        if self.interpolation:
            # performing interpolation
            self.interpolated_probs = self.linear_interpolation(l1=(1 / 3),
                                                                l2=(1 / 3),
                                                                l3=(1 / 3))

    def calculate_bayes_probs(self):
        """
        This methods calculates the Bayes probability for each ngram using
        the following equation: P(w_n | w_(n-k),..., w_(n-1)) = count(w_(n-k),.., w_n) / count(w_(n-k),..., w_(n-1))

        :return: A dictionary with the bayes probabilities for each ngram tuple.
        """
        pr = map(lambda ngram_tuple: (ngram_tuple,
                                      self.model_ngrams[ngram_tuple] / self.model_ngrams.get(ngram_tuple[:-1],
                                                                                             self.tokens_count)),
                 self.model_ngrams)

        return dict(pr)

    def perform_smoothing(self, smoothing_algo):
        """
        This method handles smoothing process and calls a certain smoothing algorithm
        based on the given name.
        :param smoothing_algo: A string with the name of the smoothing algorithm that should be performed.
        """
        if smoothing_algo == "laplace_smoothing":
            logger.info("Running Laplace smoothing process...")
            self.smoothed_probs = self.laplace_smoothing(add_k=self.laplace_k)

        elif smoothing_algo == "k_n":
            logger.info("Running Kneser-Ney smoothing process...")
            self.smoothed_probs = self.kneser_ney_smoothing(d=self.knener_ney_d)
        else:
            logger.warning("Please choose a valid smoothing method.")

    def laplace_smoothing(self, add_k=1):
        """
        This method performs add-k smoothing algorithm. By default the k is equal to 1
        and thus it performs the Laplace smoothing algorithm.
        :param add_k: int. The k param
        :return: A dictionary with the smoothed probabilities for each n-gram tuple
        """

        lp = map(lambda ngram_tuple: (ngram_tuple,
                                      (self.model_ngrams.get(ngram_tuple, 0) + add_k) /
                                      (self.model_ngrams.get(ngram_tuple[:-1],
                                                             self.tokens_count) + (add_k * len(self.vocabulary)))),
                 self.model_ngrams)

        return dict(lp)

    def kneser_ney_smoothing(self, d):
        """
        This method performs Kneser-Ney smoothing algorithm. By default the discount value is set to 0.75.
        :param d: The discount value of the algorithm.
        :return: A dictionary with the smoothed probabilities for each n-gram tuple
        """

        if self.n_model == 2:

            smoothed_probs_dict = dict()

            for ngram_t in self.model_ngrams_raw[2]:
                dict_values1 = {v for t, v in self.model_ngrams_raw[2].items() if t[0] == ngram_t[0]}

                dict_values_sum1 = sum(dict_values1)
                if dict_values_sum1 == 0:
                    dict_values_sum1 += 1

                dict_values_count1 = len(dict_values1)

                dict_values2 = {v for t, v in self.model_ngrams_raw[2].items() if t[1] == ngram_t[0]}
                dict_values_count2 = len(dict_values2)

                max = np.max(self.model_ngrams[ngram_t] - d, 0)

                l = (d * dict_values_count1) / dict_values_sum1
                k_n = (max / (self.model_ngrams[ngram_t[:-1]])) + (
                    (l * dict_values_count2) / len(self.model_ngrams_raw[2]))

                smoothed_probs_dict[ngram_t] = k_n

        elif self.n_model == 3:

            smoothed_probs_dict = dict()

            for ngram_t in self.model_ngrams_raw[3]:

                dict_values1 = {v for t, v in self.model_ngrams_raw[3].items() if t[:-1] == ngram_t[:-1]}

                dict_values_sum1 = sum(dict_values1)
                if dict_values_sum1 == 0:
                    dict_values_sum1 += 1

                dict_values_count1 = len(dict_values1)

                dict_values2 = {v for t, v in self.model_ngrams_raw[3].items() if t[-2:] == ngram_t[:-1]}
                dict_values_count2 = len(dict_values2)

                if dict_values_count2 == 0:
                    dict_values_count2 += 1

                max = np.max(self.model_ngrams[ngram_t] - d, 0)

                l = (d * dict_values_count1) / dict_values_sum1
                k_n = (max / (self.model_ngrams[ngram_t[:-1]])) + (
                    (l * dict_values_count2) / len(self.model_ngrams_raw[2]))

                smoothed_probs_dict[ngram_t] = k_n

        else:
            raise NotImplementedError('Sorry, next time!')

        return smoothed_probs_dict

    def linear_interpolation(self, l1, l2, l3):
        """
        This method performs linear interpolation to the n-gram smoothed probabilities of the model.
        :param l1: The weight l1
        :param l2: The weight l2
        :param l3: The weight l3
        :return: A dictionary with the interpolated probabilities for each n-gram tuple
        """
        assert (l1 + l2 + l3 == 1)

        intepolated_probs = dict()
        for ngram_t in self.smoothed_probs:
            intepolated_probs[ngram_t] = sum([l3 * self.smoothed_probs.get(ngram_t, 0),
                                              l2 * self.smoothed_probs.get(ngram_t[:-1], 0),
                                              l1 * self.smoothed_probs.get(ngram_t[:-2], 0)])

        return intepolated_probs

    def get_test_ngrams_smoothed_probs(self, test_ngram_tuples):
        """
        This method performs smoothing on the n-grams from the test set. In that way, the zero probabilities of the
        unknown/ new words from the test text, will have very low probability assigned to them.
        :param test_ngram_tuples: A dictionary with the n-grams of the test set.
        :return: A dictionary with the smoothed probabilities of the n-grams of the test set.
        """
        test_probs = dict()

        if self.interpolation:

            for ngram_t in test_ngram_tuples:

                if self.interpolated_probs.get(ngram_t):
                    test_probs[ngram_t] = self.interpolated_probs.get(ngram_t)

                else:
                    if self.interpolated_probs.get(ngram_t[:-1]):

                        test_probs[ngram_t] = 1 / (self.model_ngrams[ngram_t[:-1]] + len(self.vocabulary))
                    else:

                        if self.interpolated_probs.get(ngram_t[:-2]):

                            test_probs[ngram_t] = 1 / (self.model_ngrams[ngram_t[:-2]] + len(self.vocabulary))
                        else:
                            test_probs[ngram_t] = 1 / (self.tokens_count + len(self.vocabulary))
        else:

            for ngram_t in test_ngram_tuples:

                if self.smoothed_probs.get(ngram_t):
                    test_probs[ngram_t] = self.smoothed_probs.get(ngram_t)

                else:
                    if self.smoothed_probs.get(ngram_t[:-1]):

                        test_probs[ngram_t] = 1 / (self.model_ngrams[ngram_t[:-1]] + len(self.vocabulary))
                    else:

                        if self.smoothed_probs.get(ngram_t[:-2]):

                            test_probs[ngram_t] = 1 / (self.model_ngrams[ngram_t[:-2]] + len(self.vocabulary))
                        else:
                            test_probs[ngram_t] = 1 / (self.tokens_count + len(self.vocabulary))

        self.test_probs = test_probs
        return test_probs

    def compute_sum_of_probability(self, ngrams):
        """
        This method calculates the sum of log probabilities.
        :param ngrams: A list with the n-grams
        :return: The sum of log probabilities
        """
        sum_of_probs = -sum(map(np.log,
                                self.get_test_ngrams_smoothed_probs(test_ngram_tuples=ngrams).values()))

        return sum_of_probs

    def create_wrong_sentence(self, n_words, vocabulary):
        """
        This method creates a sentence of <n_word> length by randomly select words from a vocabulary.
        :param n_words: The length of the produced random sentence.
        :param vocabulary: The vocabulary pool that the sentence will be generated from.
        :return: A list of tokens that in order they create the random sentence.
        """

        tokens = list()

        for words in range(n_words):
            tokens.append(random.choice(vocabulary))

        start = ['<s{}>'.format(i) for i in range(1, self.n_model)]
        end = ['</s{}>'.format(i) for i in reversed(range(1, self.n_model))]

        return start + tokens + end

    def evaluate_on_sentences_pairs(self, sentence, vocabulary):
        """
        This method takes a random sentence from the test data, creates a random sentence with the same length
        and calculate the sum of log probabilities that our trained model assigns on them.
        :param sentence: Correct sentence from the test set.
        :param vocabulary: Vocabulary of the model.
        """
        filtered_correct = [t for t in sentence if t not in ['<s1>', '<s2>', '</s1>', '</s1>']]

        correct_sentence_len = len(filtered_correct)

        random_sentence_tokens = self.create_wrong_sentence(n_words=correct_sentence_len,
                                                            vocabulary=vocabulary)

        random_sentence_ngrams = Preprocessor.create_ngrams(seq=random_sentence_tokens,
                                                            n=self.n_model)

        correct_sentence_ngrams = Preprocessor.create_ngrams(seq=sentence,
                                                             n=self.n_model)

        filtered_random = [t for t in random_sentence_tokens if t not in ['<s1>', '<s2>', '</s1>', '</s1>']]

        random_sum_of_log_probs = self.compute_sum_of_probability(ngrams=random_sentence_ngrams)

        correct_sum_of_log_probs = self.compute_sum_of_probability(ngrams=correct_sentence_ngrams)

        logger.info('Correct Sentence: "{}."'.format(' '.join(filtered_correct).capitalize()))
        logger.info('Correct Sentence Sum of Log Probs: {}.'.format(correct_sum_of_log_probs))

        logger.info('Random Sentence: "{}."'.format(' '.join(filtered_random).capitalize()))
        logger.info('Random Sentence Sum of Log Probs: {}.'.format(random_sum_of_log_probs))

    def mle_predict_word(self, word_tuple, n_suggestions=3):
        """
        This method performs the Maximum Likelihood Estimation algorithm and finds
        the top N most likely words that will follow a given word.

        :param word_tuple: The word we want to find the next one.
        :param n_suggestions: int. The top N most probable words
        :return: A dictionary with max N ordered probabilities and their respective words
        """
        next_words = dict()
        for ngram_tuple in self.probs:
            if ngram_tuple[:-1] == word_tuple:
                ending_tuple = ngram_tuple[len(word_tuple):]

                next_words[ending_tuple] = self.probs[ngram_tuple]

        sorted_ngams = sorted(next_words.items(), key=operator.itemgetter(1), reverse=True)

        return sorted_ngams[:n_suggestions]


if __name__ == '__main__':
    print("The model is trained. Please wait for you input...")

    train_example_ngrams = {1: {('<s2>',): 4,
                                ('i',): 4,
                                ('want',): 4,
                                ('to',): 4,
                                ('eat',): 4,
                                ('chinese',): 1,
                                ('food',): 1,
                                ('lunch',): 1,
                                ('spend',): 1,
                                ('</s2>',): 4},
                            2: {('<s2>', 'i'): 4,
                                ('i', 'want'): 4,
                                ('want', 'to'): 4,
                                ('to', 'eat'): 4,
                                ('eat', 'chinese'): 1,
                                ('eat', '</s2>'): 3,
                                ('chinese', 'food'): 1,
                                ('food', 'lunch'): 1,
                                ('lunch', 'spend'): 1,
                                ('spend', '</s2>'): 1},
                            3: {('<s1>', '<s2>', 'i'): 4,
                                ('<s2>', 'i', 'want'): 4,
                                ('i', 'want', 'to'): 4,
                                ('want', 'to', 'eat'): 4,
                                ('to', 'eat', '</s2>'): 1,
                                ('to', 'eat', 'chinese'): 3,
                                ('eat', 'chinese', 'food'): 3,
                                ('chinese', 'food', 'lunch'): 3,
                                ('food', 'lunch', 'spend'): 3,
                                ('lunch', 'spend', '</s2>'): 3,
                                ('spend', '</s2>', '</s1>'): 3,
                                ('eat', '</s2>', '</s1>'): 1,
                                }}

    test_bigrams = [('<s>', 'i'),
                    ('i', 'want'),
                    ('want', 'to'),
                    ('to', 'eat'),
                    ('eat', 'greek'),
                    ('greek', 'food'),
                    ('food', '</s>')]

    test_trigrams = [('<s1>', '<s2>', 'i'),
                     ('<s2>', 'i', 'want'),
                     ('i', 'want', 'to'),
                     ('want', 'to', 'eat'),
                     ('to', 'eat', 'greek'),
                     ('eat', 'greek', 'food'),
                     ('greek', 'food', '</s2>'),
                     ('food', '</s2>', '</s1>')]

    # Create a model object with the dictionaries above
    modelObj = Model(train_example_ngrams, n_model=2, interpolation=True, kneser_ney_d=0.75)

    # Fit model to data
    modelObj.fit_model(smoothing_algo="k_n")
    print('Probs')
    pprint(modelObj.probs)
    print()
    print('Smoothed')
    pprint(modelObj.smoothed_probs)
    print()
    print('Interpolated')
    pprint(modelObj.interpolated_probs)
    print('-' * 100)

    # Predict on test data
    print('Test probs')
    modelObj.get_test_ngrams_smoothed_probs(test_ngram_tuples=test_trigrams)
    pprint(modelObj.test_probs)
    print('-' * 100)

    # Evaluate model
    eval_obj = Evaluation(model_to_test=modelObj.test_probs,
                          data_to_test=test_trigrams,
                          tokens_count=modelObj.tokens_count)
    print('Cross Entropy: {}'.format(np.round(eval_obj.compute_model_performance()['cross_entropy'], 3)))
    print('Perplexity: {}'.format(np.round(eval_obj.compute_model_performance()['perplexity'], 3)))

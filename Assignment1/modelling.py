import numpy as np
import operator
from pprint import pprint
from collections import defaultdict
from Assignment1 import setup_logger
import random
from Assignment1.preprocess import Preprocessor

logger = setup_logger(__name__)


class Model(object):
    def __init__(self, model_ngrams, n_model=2):
        """

        :param model_ngrams:
        :param test_ngram_tuples:
        :param n_model:
        """

        self.n_model = n_model

        self.model_ngrams = self.merge_ngram_counts(model_ngrams)
        self.vocabulary = model_ngrams[1]  # will always be the 1-gram counts.

        self.tokens_count = sum(self.vocabulary.values())

        self.probs = dict()
        self.test_probs = dict()
        self.smoothed_probs = dict()
        self.interpolated_probs = dict()
        self.log_probs = dict()
        self.trained_model = dict()

    @staticmethod
    def merge_ngram_counts(ngrams):
        """

        :return:
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
        :return:
        """
        assert smoothing_algo in ['laplace_smoothing']

        self.probs = self.calculate_bayes_probs()
        self.perform_smoothing(smoothing_algo)
        # self.linear_interpolation(l1=0.5, l2=0.3, l3=0.2)
        # self.log_prob()
        self.mle()

    def calculate_bayes_probs(self):
        """
        This methods calculates the Bayes probability for each ngram using
        the following equation: P(w2 | w1) = count(w1, w2) / count(w1).

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
        :param smoothing_algo: A string with the name of the smoothing algorithm.
        :return:
        """
        if smoothing_algo == "laplace_smoothing":
            logger.info("Running Laplace smoothing process...")
            self.smoothed_probs = self.laplace_smoothing()

        elif smoothing_algo == "k_n":
            logger.info("Running Kneser-Ney smoothing process...")
            self.smoothed_probs = self.kneser_ney_smoothing()

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

    def kneser_ney_smoothing(self):
        """

        :return:
        """
        return self.probs

    def linear_interpolation(self, l1, l2, l3):
        """

        :return:
        """
        assert (l1 + l2 + l3 == 1)
        self.interpolated_probs = dict(map(lambda k: (
                l1 * self.smoothed_probs[k] +
                l2 * self.smoothed_probs[k] +
                l3 * self.smoothed_probs[k]), self.smoothed_probs))

    def log_prob(self):
        """

        :return: A dictionary with the logged probabilities for each ngram tuple
        """
        self.log_probs = dict(map(lambda k: (k, - np.log(self.smoothed_probs[k])), self.smoothed_probs))

    def mle(self):
        """
        This method performs the Maximum Likelihood Estimation algorithm to the ngrams dictionary
        and  keeps the ngram with the max prob for each word.
        :return: A dictionary with the maximum probability for each word
        """
        for word in self.vocabulary.keys():
            ngram_to_store = None
            max_value = None
            for ngram in self.smoothed_probs.keys():
                if word == ngram[0]:
                    if max_value is None:
                        ngram_to_store = ngram
                        max_value = self.smoothed_probs[ngram]

                    elif max_value < self.smoothed_probs[ngram]:
                        ngram_to_store = ngram
                        max_value = self.smoothed_probs[ngram]

            self.trained_model[ngram_to_store] = max_value

    def mle_predict_word(self, word_tuple, n_suggestions=3):
        """
        This method performs the Maximum Likelihood Estimation algorithm and finds
        the top N most likely words that will follow a given word.

        :param word_tuple: The word we want to find the next one.
        :param n_suggestions: int. The top N most probable words
        :return: A dictionary with max 3 ordered probabilities and their respective words
        """
        next_words = dict()
        for ngram_tuple in self.probs:
            if ngram_tuple[:-1] == word_tuple:
                ending_tuple = ngram_tuple[len(word_tuple):]

                next_words[ending_tuple] = self.probs[ngram_tuple]

        sorted_ngams = sorted(next_words.items(), key=operator.itemgetter(1), reverse=True)

        return sorted_ngams[:n_suggestions]

    def get_test_ngrams_smoothed_probs(self, test_ngram_tuples):
        """

        :param test_ngram_tuples:
        :return:
        """
        test_probs = dict()

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

        :param ngrams:
        :return:
        """
        sum_of_probs = -sum(map(np.log,
                                self.get_test_ngrams_smoothed_probs(test_ngram_tuples=ngrams).values()))

        return sum_of_probs

    def create_wrong_sentence(self, n_words, vocabulary):
        """

        :param n_words:
        :param vocabulary:
        :param n_model:
        :return:
        """

        tokens = list()

        for words in range(n_words):
            tokens.append(random.choice(vocabulary))

        start = ['<s{}>'.format(i) for i in range(1, self.n_model)]
        end = ['</s{}>'.format(i) for i in reversed(range(1, self.n_model))]

        return start + tokens + end

    def evaluate_on_sentences_pairs(self, sentence, vocabulary):
        """

        :param sentences:
        :param vocabulary:
        :return:
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


if __name__ == '__main__':
    print("The model is trained. Please wait for you input...")
    # Test case with the following dictionaries
    tokens = ["<s>", "i", "want", "to", "eat", "chinese", "food", "lunch", "spend", "</s>",
              "<s>", "i", "want", "to", "eat", "chinese", "food", "lunch", "spend", "</s>",
              "<s>", "i", "want", "to", "eat", "chinese", "food", "lunch", "spend", "</s>",
              "<s>", "i", "want", "to", "eat", "</s>"]

    vocabulary_freq = {"i": 4,
                       "want": 4,
                       "to": 4,
                       "eat": 4,
                       "chinese": 3,
                       "food": 3,
                       "lunch": 3,
                       "spend": 3,
                       "<s>": 4,
                       "</s>": 4}

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
                            2:
                                {('<s2>', 'i'): 4,
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

                                }
                            }

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
    modelObj = Model(train_example_ngrams, n_model=3)

    # fit model to data
    modelObj.fit_model("laplace_smoothing")
    # print()
    # print('Probs')
    # pprint(modelObj.probs)
    # print()
    # print('Smoothed')
    # pprint(modelObj.smoothed_probs)
    # print()
    #
    # print('Checking test probs')
    # modelObj.get_test_ngrams_smoothed_probs(test_ngram_tuples=test_trigrams)
    # pprint(modelObj.test_probs)

    print("Model have been trained!")

    word = input("Please type a word \n")

    # predict
    mle_dict = modelObj.mle_predict_word((word,))
    print(mle_dict)

    from Assignment1.evaluation import Evaluation

    eval_obj = Evaluation(model_to_test=modelObj.test_probs,
                          data_to_test=test_trigrams,
                          tokens_count=modelObj.tokens_count)

    pprint(eval_obj.compute_model_performance())

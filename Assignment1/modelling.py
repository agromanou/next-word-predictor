import numpy as np
import operator
from pprint import pprint


class Model(object):
    def __init__(self, ngrams):

        self.ngrams = ngrams[2]
        self.vocabulary = ngrams[1]
        self.tokens_count = sum(self.vocabulary.values())
        self.probs = dict()
        self.smoothed_probs = dict()
        self.interpolated_probs = dict()
        self.log_probs = dict()
        self.trained_model = dict()

    def fit_model(self, smoothing_algo):
        """
        This method runs the modeling process by calculating the Bayes probabilities
        and performing smoothing on the Models variables.
        :param smoothing_algo: The name of the smoothing algorithm that will be used.
        :return:
        """
        assert smoothing_algo in ['laplace_smoothing', 'k_n']

        self.probs = self.calculate_bayes_probs()
        self.perform_smoothing(smoothing_algo)
        self.linear_interpolation(l1=0.5, l2=0.3, l3=0.2)
        self.log_prob()
        self.mle()

    def calculate_bayes_probs(self):
        """
        This methods calculates the Bayes probability for each ngram using
        the following equation: P(w2 | w1) = count(w1, w2) / count(w1).
        :param grams: A dictionary with pairs of words and their frequencies
        :param voc: A dictionary with each unique word and their frequencies.
        :return: A dictionary with the bayes probabilities for each ngram tuple.
        """
        return dict(
            map(lambda ngram_tuple: (ngram_tuple, self.ngrams[ngram_tuple] / self.vocabulary[(ngram_tuple[0],)]),
                self.ngrams))

    def perform_smoothing(self, smoothing_algo):
        """
        This method handles smoothing process and calls a certain smoothing algorithm
        based on the given name.
        :param smoothing_algo: A string with the name of the smoothing algorithm.
        :return:
        """
        if smoothing_algo == "laplace_smoothing":
            print("Running Laplace smoothing process..")
            self.smoothed_probs = self.laplace_smoothing()

        elif smoothing_algo == "k_n":
            print("Running Kneser-Ney smoothing process..")
            self.smoothed_probs = self.kneser_ney_smoothing()

        else:
            print("Please choose a valid smoothing method.")

    def laplace_smoothing(self, add_k=1):
        """
        This method performs add-k smoothing algorithm. Be default the k is equal to 1
        and thus it performs the Laplace smoothing algorithm.
        :param add_k: The k param
        :return: A dictionary with the smoothed probabilities for each ngram tuple
        """
        pl = dict(
            map(lambda ngram_tuple: (ngram_tuple, (self.ngrams[ngram_tuple] + add_k) /
                                     (self.vocabulary[(ngram_tuple[0],)] + len(self.vocabulary))), self.ngrams))

        return pl

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

    def mle_predict_word(self, word):
        """
        This method performs the Maximum Likelihood Estimation algorithm and finds
        the 3 most likely words that will follow a given word.
        :param word: The word we want to find the next one.
        :return: A dictionary with max 3 ordered probabilities and their respective words
        """
        next_words = {}
        for k in self.smoothed_probs.keys():
            if k[0] == word:
                next_words[k[1]] = self.smoothed_probs[k]

        sorted_ngams = sorted(next_words.items(), key=operator.itemgetter(1), reverse=True)

        return sorted_ngams


if __name__ == '__main__':
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

    ngrams = {1: {('<s>',): 4,
                  ('i',): 4,
                  ('want',): 4,
                  ('to',): 4,
                  ('eat',): 4,
                  ('chinese',): 1,
                  ('food',): 1,
                  ('lunch',): 1,
                  ('spend',): 1,
                  ('</s>',): 4},
              2:
                  {('<s>', 'i'): 4,
                   ('i', 'want'): 4,
                   ('want', 'to'): 4,
                   ('to', 'eat'): 4,
                   ('eat', 'chinese'): 1,
                   ('eat', '</s>'): 3,
                   ('chinese', 'food'): 1,
                   ('food', 'lunch'): 1,
                   ('lunch', 'spend'): 1,
                   ('spend', '</s>'): 1}
              }

    # Create a model object with the dictionaries above
    modelObj = Model(ngrams)

    # fit model to data
    modelObj.fit_model("laplace_smoothing")

    pprint(modelObj.probs)
    print()
    pprint(modelObj.smoothed_probs)
    print()
    pprint(modelObj.log_probs)

    # predict
    mle_dict = modelObj.mle_predict_word("eat")
    print(mle_dict)

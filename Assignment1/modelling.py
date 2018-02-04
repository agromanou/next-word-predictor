import numpy as np
import operator


class Model(object):
    def __init__(self,
                 vocabulary,
                 tokens,
                 ngrams):

        self.vocabulary = vocabulary
        self.tokens = tokens
        self.ngrams = ngrams
        self.probs = {}
        self.smoothed_probs = {}
        self.log_probs = {}
        self.trained_model = {}

    def fit_model(self, smoothing_algo):
        """
        This method runs the modeling process by calculating the Bayes probabilities
        and performing smoothing on the Models variables.
        :param smoothing_algo: The name of the smoothing algorithm that will be used.
        :return:
        """
        assert smoothing_algo in ['laplace_smoothing', 'k_n']

        self.probs = self.calculate_bayes_probs(self.ngrams, self.vocabulary)
        self.perform_smoothing(smoothing_algo)
        self.log_prob()
        self.mle()

        print(self.vocabulary)
        print(len(self.vocabulary))
        print(self.ngrams)
        print(self.probs)
        print(self.smoothed_probs)
        print(self.log_probs)
        print(self.trained_model)

    @staticmethod
    def calculate_bayes_probs(grams, voc):
        """
        This methods calculates the Bayes probability for each ngram using
        the following equation: P(w2 | w1) = count(w1, w2) / count(w1).
        :param grams: A dictionary with pairs of words and their frequencies
        :param voc: A dictionary with each unique word and their frequencies.
        :return: A dictionary with the bayes probabilities for each ngram tuple.
        """
        return dict(map(lambda p: (p, grams[p] / voc[p[0]]), grams))

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
            map(lambda c: (c[0], (c[1] + add_k) / (len(self.tokens) + add_k * len(self.vocabulary))),
                self.probs.items()))
        return pl

    def kneser_ney_smoothing(self):
        """

        :return:
        """
        return self.probs

    def log_prob(self):
        """

        :return: A dictionary with the logged probabilities for each ngram tuple
        """
        log_prob = dict(map(lambda k: (k, - np.log(self.smoothed_probs[k])), self.smoothed_probs))
        self.log_probs = log_prob

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
        the to 3 most likely words that will follow a given word.
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

    ngrams = {('<s>', 'i'): 4,
              ('i', 'want'): 4,
              ('want', 'to'): 4,
              ('to', 'eat'): 4,
              ('eat', 'chinese'): 3,
              ('eat', '</s>'): 1,
              ('chinese', 'food'): 3,
              ('food', 'lunch'): 3,
              ('lunch', 'spend'): 3,
              ('spend', '</s>'): 3}

    # Create a model object with the dictionaries above
    modelObj = Model(vocabulary_freq,
                     tokens,
                     ngrams)

    # fit model to data
    modelObj.fit_model("laplace_smoothing")

    # predict
    mle_dict = modelObj.mle_predict_word("eat")
    print(mle_dict)

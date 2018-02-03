import numpy as np

class Model(object):
    def __init__(self,
                 smoothing_algo,
                 vocabulary,
                 tokens):

        assert smoothing_algo in ['laplace_smoothing', 'k_n']

        self.smoothing_algo = smoothing_algo
        self.vocabulary = vocabulary
        self.tokens = tokens
        self.prob = {}

    @staticmethod
    def calculate_bayes_probs(grams, voc):
        """
        P(w2 | w1) = count(w1, w2) / count(w1)
        :param ngrams:
        :param tokens:
        :return:
        """
        return dict(map(lambda p: (p, grams[p] / voc[p.split()[0]]), grams))

    def perform_smoothing(self):
        """

        :return:
        """
        if self.smoothing_algo == "laplace_smoothing":

            print("Running Laplace smoothing process..")
            self.prob = self.laplace_smoothing()

        elif self.smoothing_algo == "k_n":

            print("Running Kneser-Ney smoothing process..")
            self.prob = self.kneser_ney_smoothing()
        else:
            print("Please choose a valid smoothing method.")

    def laplace_smoothing(self, add_k=1):
        """

        :param add_k: Int.
        :return:
        """
        pl = dict(
            map(lambda c: (c[0], (c[1] + add_k) / (len(self.tokens) + add_k * len(self.vocabulary))),
                      self.vocabulary.items()))
        return pl

    def kneser_ney_smoothing(self):
        """

        :return:
        """
        pass

    def log_prob(self):
        """

        :return: a dictionary with n-grams and assigned log probabilities
        """
        log_prob = dict(map(lambda k: (k, np.log(self.prob[k])), self.prob))
        return log_prob

    def mle(self):
        pass


if __name__ == '__main__':

    tokens = ["i", "want", "to", "eat", "chinese", "food", "lunch", "spend",
              "i", "want", "to", "eat", "chinese", "food", "lunch", "spend",
              "i", "want", "to", "eat", "chinese", "food", "lunch", "spend",
              "i", "want", "to", "eat"]

    vocabulary_freq = {"i": 4,
                       "want": 4,
                       "to": 4,
                       "eat": 4,
                       "chinese": 3,
                       "food": 3,
                       "lunch": 3,
                       "spend": 3}

    ngrams = {'i want': 4,
              'want to': 4,
              'to eat': 4,
              'eat chinese': 3,
              'chinese food': 3,
              'food lunch': 3,
              'lunch spend': 3}

    modelObj = Model("laplace_smoothing",
                     vocabulary_freq,
                     tokens)
    modelObj.perform_smoothing()
    print(modelObj.prob)

    probs = Model("laplace_smoothing",
                  vocabulary_freq,
                  tokens).calculate_bayes_probs(ngrams,
                                                vocabulary_freq)
    print(probs)

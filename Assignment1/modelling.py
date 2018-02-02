class Model(object):
    def __init__(self, smoothing_algo, vocabulary, tokens):
        self.smoothing_algo = smoothing_algo
        self.vocabulary = vocabulary
        self.tokens = tokens
        self.prob = {}

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

        :param add_k:
        :return:
        """
        pl = dict(map(lambda c: (c[0], (c[1] + add_k) / (len(self.tokens) + add_k * len(self.vocabulary))),
                      self.vocabulary.items()))
        return pl

    def kneser_ney_smoothing(self):
        """

        :return:
        """
        pass


if __name__ == '__main__':
    vocabulary_freq = {"i": 4,
                       "want": 4,
                       "to": 4,
                       "eat": 4,
                       "chinese": 4,
                       "food": 4,
                       "lunch": 4,
                       "spend": 4}

    tokens = ["i", "want", "to", "eat", "chinese", "food", "lunch", "spend",
              "i", "want", "to", "eat", "chinese", "food", "lunch", "spend",
              "i", "want", "to", "eat", "chinese", "food", "lunch", "spend",
              "i", "want", "to", "eat", "chinese", "food", "lunch", "spend"]

    modelObj = Model("laplace_smoothing", vocabulary_freq, tokens)
    modelObj.perform_smoothing()
    print(modelObj.prob)


class Model(object):

    def __init__(self, smoothing_algo):
        self.smoothing_algo = smoothing_algo
        self.word_counts = []
        self.vocabulary = []
        self.tokens = []

    def perform_smoothing(self):
        """

        :return:
        """
        if self.smoothing_algo == "":
            self.laplace_smoothing()
        elif self.smoothing_algo == "":
            self.kneser_ney_smoothing()
        else:
            print("Please choose a valid smoothing method.")

    def laplace_smoothing(self, add_k=1):
        """

        :param add_k:
        :return:
        """
        pl = list(map(lambda c: (c + add_k) / (len(self.tokens) + add_k * len(self.vocabulary)), self.word_counts))

        return pl

    def kneser_ney_smoothing(self):
        """

        :return:
        """
        pass


if __name__ == '__main__':
    pass

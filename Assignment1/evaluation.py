import numpy as np
from pprint import pprint


class Evaluation(object):

    def __init__(self, model_to_test, data_to_test, tokens_count):
        """
        This class is responsible for the evaluation of the language model.
        :param model_to_test:
        :param data_to_test:
        """
        self.tokens_count = tokens_count
        self.model = model_to_test
        self.data_to_test = data_to_test

        self.cross_entropy = None
        self.perplexity = None

    def compute_model_performance(self):
        """

        :return:
        """
        cross_entropy = self.compute_cross_entropy()
        perplexity = self.compute_perplexity()

        self.cross_entropy = cross_entropy
        self.perplexity = perplexity

        return dict(cross_entropy=cross_entropy,
                    perplexity=perplexity)

    def compute_perplexity(self):
        """
        This method computes the perplexity for a given test dataset.

        :return:
        """

        total = 0
        for n_gram in self.data_to_test:
            total = total + (- np.log(self.model.get(n_gram, 1)))

        return np.power(total, - (1 / self.tokens_count))

    def compute_cross_entropy(self):
        """
        This method computes the cross entropy for a given test dataset.

        :return:
        """
        total = 0
        for n_gram in self.data_to_test:
            # pprint(n_gram)
            # pprint(self.model.get(n_gram, 'missing'))
            # vasoume 1 gia na ginei log1 = 0
            total = total + np.log(self.model.get(n_gram, 1))

        return - (total / self.tokens_count)


if __name__ == '__main__':
    tokens_test = ["<s>", "i", "want", "to", "eat", "chinese", "food", "lunch", "spend", "</s>",
                   "<s>", "i", "want", "to", "eat", "chinese", "food", "lunch", "spend", "</s>",
                   "<s>", "i", "want", "to", "eat", "chinese", "food", "lunch", "spend", "</s>",
                   "<s>", "i", "want", "to", "eat", "</s>"]

    ngrams_test = {('<s>', 'i'): 4,
                   ('i', 'want'): 4,
                   ('want', 'to'): 4,
                   ('to', 'eat'): 4,
                   ('eat', 'chinese'): 3,
                   ('eat', '</s>'): 1,
                   ('chinese', 'food'): 3,
                   ('food', 'lunch'): 3,
                   ('lunch', 'spend'): 3,
                   ('spend', '</s>'): 3}

    model = {('<s>', 'i'): 0.043478260869565216,
             ('chinese', 'food'): 0.043478260869565216,
             ('to', 'eat'): 0.043478260869565216,
             ('lunch', 'spend'): 0.043478260869565216,
             ('spend', '</s>'): 0.043478260869565216,
             ('eat', 'chinese'): 0.03804347826086957,
             ('eat', '</s>'): 0.02717391304347826,
             ('want', 'to'): 0.043478260869565216,
             ('i', 'want'): 0.043478260869565216,
             ('food', 'lunch'): 0.043478260869565216}

    eval_obj = Evaluation(model, ngrams_test)
    print(eval_obj.compute_model_performance())

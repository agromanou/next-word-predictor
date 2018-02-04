import numpy as np


class Evaluation(object):
    def __init__(self, model, data_to_test):
        self.model = model
        self.data_to_test = data_to_test

    def compute_model_performance(self):
        """

        :return:
        """
        cross_entropy = self.compute_cross_entropy()
        perplexity = self.compute_perplexity()

        return cross_entropy, perplexity

    def compute_perplexity(self):
        """

        :return:
        """
        total = 0
        for ngram in self.data_to_test.keys():
            total = total + (- np.log(self.model[ngram]))

        print(total, len(self.data_to_test), 1 / len(self.data_to_test))

        return np.power(total, - (1 / len(self.data_to_test)))

    def compute_cross_entropy(self):
        """
        
        :return:
        """
        total = 0
        for ngram in self.data_to_test.keys():
            total = total + np.log2(self.model[ngram])

        return - (total / len(self.data_to_test))

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
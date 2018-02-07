from Assignment1.modelling import Model

import unittest


class ModelTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @staticmethod
    def test_laplace_smoothing_normal_execution():
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

        exp_outcome = {"i": 0.125,
                       "want": 0.125,
                       "to": 0.125,
                       "eat": 0.125,
                       "chinese": 0.125,
                       "food": 0.125,
                       "lunch": 0.125,
                       "spend": 0.125}

        model_obj = Model("laplace_smoothing", vocabulary_freq, tokens)
        pl = model_obj.laplace_smoothing()

        assert (pl == exp_outcome)

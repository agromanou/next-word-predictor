from Assignment1.text_utils import TextUtils

import unittest
import pandas as pd
import numpy as np


class TextUtilsTest(unittest.TestCase):

    def setUp(self):
        self.theTextUnitObject = TextUtils()

    def tearDown(self):
        pass

    def test_create_ngrams_with_2_n(self):
        sentence = "This is a quite large sentence"
        exp_outcome = [['This', 'is'],
                   ['is', 'a'],
                   ['a', 'quite'],
                   ['quite', 'large'],
                   ['large', 'sentence']]
        print(sentence)

        ngrams = self.theTextUnitObject.create_ngrams(sentence.split(), 2)

        for ngram in self.theTextUnitObject.create_ngrams(sentence.split(), 2):
            print(ngram)

        self.assertEqual(len(ngrams), 5)
        self.assertEqual(type(ngrams), type(exp_outcome))
        self.assertListEqual(ngrams, exp_outcome)






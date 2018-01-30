from Assignment1.text_utils import TextUtils

import unittest


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

        self.assertEqual(len(ngrams), 5)
        self.assertEqual(type(ngrams), type(exp_outcome))
        self.assertListEqual(ngrams, exp_outcome)

    def test_create_ngrams_with_3_n(self):
        sentence = "This is a quite large sentence"
        exp_outcome = [['This', 'is', 'a'],
                       ['is', 'a', 'quite'],
                       ['a', 'quite', 'large'],
                       ['quite', 'large', 'sentence']]
        print(sentence)

        ngrams = self.theTextUnitObject.create_ngrams(sentence.split(), 3)

        self.assertEqual(len(ngrams), 4)
        self.assertEqual(type(ngrams), type(exp_outcome))
        self.assertListEqual(ngrams, exp_outcome)

    def test_create_ngram_with_invalid_n(self):
        sentence = ""
        self.assertRaises(AssertionError, self.theTextUnitObject.create_ngrams, sentence.split, 1)
        self.assertRaises(AssertionError, self.theTextUnitObject.create_ngrams, sentence.split, 5)

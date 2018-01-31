from collections import Counter


class Preprocessor(object):

    def __init__(self):
        self.vocabulary = None

    @staticmethod
    def split_to_sentences(data):
        """

        :param data:
        :return:
        """
        sentences = list(map(lambda s: s.strip(), data.split('.')))
        filtered = list(filter(lambda x: x != '', sentences))

        return filtered

    @staticmethod
    def tokenize_and_pad(sentence, model_type='simple'):
        """
        This function splits a sentence into tokens. Padding is added if necessary according the model type.
        :param sentence: str.
        :param model_type: str. Enum of bigram, trigram, simple
        :return: list. An iterable of word tokens.
        """
        assert model_type in ['bigram', 'trigram', 'simple']

        words = sentence.split()

        if model_type == 'bigram':
            return ['<s1>'] + words + ['</s1>']

        elif model_type == 'trigram':
            return ['<s1>', '<s2>'] + words + ['</s1>', '</s2>']

        return words

    def create_vocabulary(self, sentences, base_limit=2):
        """
        This method counts all the tokens from a list of sentences. Then it creates a vocabulary with the most common
        tokens, that surpass the base limit.
        :param sentences: list. A list of strings.
        :param base_limit: int. A number defining the base limit for the validity of the tokens.
        :return: dict. A dictionary with the vocabulary and the rejected tokens
        """
        # grab all the tokens in an iterator. Not in a list.
        tokens = (token for sentence in sentences for token in self.tokenize_and_pad(sentence, model_type='simple'))

        tokens_count = Counter(tokens)

        valid_tokens = {k: v for k, v in tokens_count.items() if v > base_limit}
        invalid_tokens = {k: v for k, v in tokens_count.items() if v <= base_limit}

        return dict(vocabulary=valid_tokens,
                    rejected=invalid_tokens)

    @staticmethod
    def create_ngrams(seq, n):
        """
        This method creates a list of ngrams from a given sentence.

        :param seq: The given sentence.
        :param n: The length of word tuples
        :return: The n-grams for a given sentence.
        """
        assert n in [2, 3]

        return [seq[i:i + n] for i in range(len(seq) - n + 1)]


if __name__ == '__main__':
    pass
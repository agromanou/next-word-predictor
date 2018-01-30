from collections import Counter


class TextUtils:

    def __init__(self):
        pass

    @staticmethod
    def split_to_sentences(data):
        """

        :param data:
        :return:
        """
        sentences = map(lambda s: s.strip(), data.split('.'))

        return list(sentences)

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
            return ['Start1'] + words + ['End1']

        elif model_type == 'trigram':
            return ['Start1', 'Start2'] + words + ['End1', 'End2']

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

        :param seq:
        :param n:
        :return:
        """

        assert n in [2, 3]

        if n == 2:
            for num in range(1, len(seq)):
                yield (seq[num - 1], seq[num])

        elif n == 3:
            for num in range(2, len(seq)):
                yield (seq[num - 2], seq[num - 1], seq[num])

        else:
            raise Exception('N is invalid. Choose 2 or 3')

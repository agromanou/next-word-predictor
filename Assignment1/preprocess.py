from collections import Counter
from pprint import pprint
from Assignment1 import setup_logger

logger = setup_logger(__name__)
logger.disabled = True


class Preprocessor(object):

    def __init__(self):
        self.vocabulary = None

    @staticmethod
    def split_to_sentences(corpus):
        """
        This method splits a corpus into sentences.
        :param corpus: str. A textual corpus.
        :return: List. An iterable of sentences (strings).
        """
        logger.info('Splitting Corpus into sentences')

        # splitting the corpus in sentences,
        # and getting rid of the white spaces at the start and end of each sentence
        sentences = list(map(lambda s: s.strip(), corpus.split('.')))
        # filtering empty sentences
        filtered = list(filter(None, sentences))

        logger.info('Number of sentences extracted: {}'.format(len(filtered)))

        return filtered

    @staticmethod
    def flatten_to_one_corpus(sentences):
        """

        :param sentences:
        :return:
        """
        # Functional
        def flatten_iter(i, s, c, l):
            if i < l:
                c = c + ". " + s[i]
                flatten_iter(i+1, s, c, l)
            return c

        # Imperative
        # corpus = ""
        # for sentence in sentences:
        #     corpus = corpus + ". " + sentence

        return flatten_iter(0, sentences, "", len(sentences))

    @staticmethod
    def tokenize_and_pad(sentence, model_type='simple'):
        """
        This method splits a sentence into tokens. Padding is added if necessary according the model type.
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

    @staticmethod
    def create_vocabulary(tokens, base_limit=10):
        """
        This method counts all the tokens from a list of sentences. Then it creates a vocabulary with the most common
        tokens, that surpass the base limit and a rejection vocabulary for the rest.

        :param tokens: list. A list of strings.
        :param base_limit: int. A number defining the base limit for the validity of the tokens.
        :return: dict. A dictionary with the vocabulary and the rejected tokens
        """

        logger.info('Creating Vocabulary with base_limit: {}'.format(base_limit))

        # # grab all the tokens in an iterator. Not in a list.
        # tokens = (token for sentence in sentences for token in self.tokenize_and_pad(sentence, model_type='simple'))

        tokens_count = Counter(tokens)

        # selecting the valid tokens, and the rejected tokens in separate dictionaries.
        valid_tokens = {k: v for k, v in tokens_count.items() if v > base_limit}
        invalid_tokens = {k: v for k, v in tokens_count.items() if v <= base_limit}

        logger.info('Valid Vocabulary size: {}'.format(len(valid_tokens)))
        logger.info('Rejection Vocabulary size: {}'.format(len(invalid_tokens)))

        return dict(vocabulary=valid_tokens, rejected=invalid_tokens)

    @staticmethod
    def create_ngrams(seq, n):
        """
        This method creates a list of ngrams from a given sentence.

        :param seq: The given sentence.
        :param n: The length of word tuples
        :return: The n-grams for a given sentence.
        """
        assert n in [1, 2, 3]

        return [seq[i:i + n] for i in range(len(seq) - n + 1)]

    def calculate_ngram_counts(self, corpus, model):
        """
        This method calculates all the n-gram counts for a given model.

        :param corpus: Str. A text
        :param model: Str. Enum defining the model that we want to use.
        :return: Counter. A python Counter of the word n-grams.
        """

        assert model in ['bigram', 'trigram']

        model_n = 2 if model == 'bigram' else 3

        all_ngrams = list()

        # splitting corpus into sentences
        sentences = self.split_to_sentences(corpus)

        # padding each sentence separately in respect to the given model
        logger.info('Padding each sentence separately in respect to the {} model'.format(model))
        padded_sentences = tuple(map(lambda s: self.tokenize_and_pad(s, model_type=model), sentences))

        # calculates all the n-grams up to the models actual n.
        # E.g for trigram, creates uni-grams, bi-grams and tri-grams
        for num in range(1, model_n + 1):

            logger.info("Creating {}-gram tokens for all padded sentences".format(num))
            sentences_ngrams = map(lambda s: self.create_ngrams(s, n=num), padded_sentences)

            # appends each n-gram into a list in order to count them.
            for sublist in sentences_ngrams:
                for item in sublist:
                    all_ngrams.append(" ".join(item))

        logger.info('Counting all n-grams')

        # counting all the different n-grams.
        counts = Counter(all_ngrams)

        logger.info('Total n-gram tokens created: {}'.format(len(counts)))

        return counts


if __name__ == '__main__':

    a_corpus = "The Cape sparrow (Passer melanurus) is a southern African bird. A medium-sized sparrow at 14–16 " \
               "centimetres (5.5–6.3 in), it has distinctive grey, brown, and chestnut plumage, with large pale " \
               "head stripes in both sexes. The male has some bold black and white markings on its head and neck." \
               " The species inhabits semi-arid savannah, cultivated areas, and towns, from the central coast of " \
               "Angola to eastern South Africa and Swaziland. Cape sparrows primarily eat seeds, along with soft " \
               "plant parts and insects. They typically breed in colonies, and search for food in large nomadic " \
               "flocks. The nest can be constructed in a tree, bush, cavity, or disused nest of another species. " \
               "A typical clutch contains three or four eggs, and both parents are involved, from nest building" \
               " to feeding the young. The species is common in most of its range and coexists successfully in urban" \
               " habitats with two of its relatives, the native southern grey-headed sparrow and the house sparrow," \
               " an introduced species. The Cape sparrow's population has not decreased significantly, and is not " \
               "seriously threatened by human activities. "

    counts = Preprocessor().calculate_ngram_counts(corpus=a_corpus, model='bigram')

    pprint(counts)

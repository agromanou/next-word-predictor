from collections import Counter
from pprint import pprint
from Assignment1 import setup_logger
from nltk.tokenize import RegexpTokenizer

import re

logger = setup_logger(__name__)


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

        # splitting the corpus in sentences, and getting rid of the white spaces at the start and end of each sentence
        sentences = list(map(lambda s: s.strip().lower(), re.split('\n', corpus)))
        # filtering empty sentences
        filtered = list(filter(None, sentences))

        logger.info('Number of sentences extracted: {}'.format(len(filtered)))

        return filtered

    @staticmethod
    def flatten_ton_one_corpus(sentences):
        """

        :param sentences:
        :return:
        """
        return '. '.join(sentences)

    @staticmethod
    def tokenize_and_pad(sentence, model_type='simple'):
        """
        This method splits a sentence into tokens. Padding is added if necessary according the model type.

        :param sentence: str.
        :param model_type: str. Enum of bigram, trigram, simple

        :return: list. An iterable of word tokens.
        """
        assert model_type in ['bigram', 'trigram', 'simple']

        mapper = {'simple': 2,
                  'bigram': 2,
                  'trigram': 3}

        sentence = re.sub('^[^a-zA-z]*|[^a-zA-Z]*$', '', sentence)

        if model_type in ['simple', 'bigram']:
            start = ['<s2>']
            end = ['</s2>']
        else:
            start = ['<s1>', '<s2>']
            end = ['</s2>', '</s1>']

        # start = ['<s{}>'.format(i) for i in range(1, mapper.get(model_type))]
        # end = ['</s{}>'.format(i) for i in reversed(range(1, mapper.get(model_type)))]

        return start + sentence.split() + end

    @staticmethod
    def create_vocabulary(tokens, base_limit=0):
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

        return valid_tokens, invalid_tokens

    @staticmethod
    def replace_uncommon_words(corpus, words, replacement='<UNK>'):
        """

        :param corpus:
        :param words:
        :param replacement:
        :return:
        """

        altered_corpus = corpus

        for w in words:
            altered_corpus = re.sub(r'\b({})\b'.format(w),
                                    replacement,
                                    altered_corpus)

        return altered_corpus

    @staticmethod
    def tokenize_corpus(corpus):
        """

        :param corpus:
        :return:
        """
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(corpus.lower())

        return tokens

    @staticmethod
    def create_ngrams(seq, n):
        """
        This method creates a list of n-grams from a given sentence.

        :param seq: The given sentence.
        :param n: The length of word tuples
        :return: The n-grams for a given sentence.
        """
        assert n in [1, 2, 3]

        return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]

    def create_ngram_metadata(self, model, list_of_tokens, threshold):
        """

        :param model:
        :param list_of_tokens:
        :param threshold:
        :return:
        """
        assert model in ['bigram', 'trigram']
        logger.info('Running Model: {}'.format(model.title()))

        model_n = 2 if model == 'bigram' else 3
        all_ngrams = dict()

        # calculates all the n-grams up to the models actual n.
        # E.g for trigram, creates uni-grams, bi-grams and tri-grams

        for num in range(1, model_n + 1):

            all_ngrams[num] = list()

            logger.info("Creating {}-gram tokens for all padded sentences".format(num))
            sentences_ngrams = map(lambda s: self.create_ngrams(s, n=num), list_of_tokens)

            # appends each n-gram into a list in order to count them.
            for sublist in sentences_ngrams:
                for item in sublist:
                    all_ngrams[num].append(item)

        logger.info('Counting all n-grams')
        counts = {}
        # counting all the different n-grams.
        for key in all_ngrams:
            counts[key] = dict(Counter(all_ngrams[key]))
            logger.info('Number of {}-grams: {}'.format(key, len(counts[key])))

        # selecting the rejected tokens
        rejected_tokens = {k[0] for k, v in counts[1].items() if v < threshold}
        logger.info('Rejection Vocabulary size: {}'.format(len(rejected_tokens)))

        final_counts = dict({i: {} for i in range(1, model_n + 1)})

        replacement="<UNK>"

        logger.info('Replacing Rejection words with {}'.format(replacement))

        for n in counts:
            for key_tuple, ngram_count in counts[n].items():
                new_tuple = tuple()
                for word in key_tuple:
                    if word in rejected_tokens:
                        new_tuple += replacement,
                    else:
                        new_tuple += word,

                final_counts[n][new_tuple] = final_counts[n].get(key_tuple, 0) + ngram_count

        del counts

        return final_counts, rejected_tokens


if __name__ == '__main__':
    a_corpus = "The Cape sparrow (Passer melanurus) is a southern African bird. \n" \
               "A medium-sized sparrow at 14–16 centimetres (5.5–6.3 in), it has distinctive grey, brown, and" \
               " chestnut plumage, with large pale head stripes in both sexes. \n " \
               "The male has some bold black and white markings on its head and neck. \n\n " \
               "The species inhabits semi-arid savannah, cultivated areas, and towns, from the central coast of " \
               "Angola to eastern South Africa and Swaziland. \n\n " \
               "Cape sparrows primarily eat seeds, along with soft plant parts and insects. They typically breed in" \
               " colonies, and search for food in large nomadic flocks. \n\n " \
               "The nest can be constructed in a tree, bush, cavity, or disused nest of another species. \n\n" \
               "A typical clutch contains three or four eggs, and both parents are involved, from nest building" \
               " to feeding the young. \n\n " \
               "The species is common in most of its range and coexists successfully in urban" \
               " habitats with two of its relatives, the native southern grey-headed sparrow and the house sparrow," \
               " an introduced species. \n\n " \
               "The Cape sparrow's population has not decreased significantly, and is not " \
               "seriously threatened by human activities. "

    test_counts = Preprocessor().tokenize_and_pad(sentence=a_corpus, model_type='bigram')

    pprint(test_counts)

from collections import Counter
from pprint import pprint
from Assignment1 import setup_logger
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import re

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
        sentences = list(map(lambda s: s.strip(), re.split('\n', corpus)))
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
                flatten_iter(i + 1, s, c, l)
            return c

        # Imperative
        # corpus = ""
        # for sentence in sentences:
        #     corpus = corpus + ". " + sentence

        return flatten_iter(0, sentences, "", len(sentences))

    @staticmethod
    def flatten_ton_one_corpus2(sentences):
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

        mapper = {'simple': 1,
                  'bigram': 2,
                  'trigram': 3}

        start = ['<s{}>'.format(i) for i in range(1, mapper.get(model_type))]
        end = ['</s{}>'.format(i) for i in range(1, mapper.get(model_type))]

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
                    all_ngrams.append(tuple(item) + ('<pad>',) * (model_n - num))

        logger.info('Counting all n-grams')

        # counting all the different n-grams.
        counts = Counter(all_ngrams)

        logger.info('Total n-gram tokens created: {}'.format(len(counts)))

        return counts, padded_sentences

    @staticmethod
    def replace_uncommon_words(corpus, words, replacement='<UNK>'):
        """

        :param corpus:
        :param words:
        :param replacement:
        :return:
        """

        altered_corpus = corpus

        replacement = " {} ".format(replacement)

        for w in words:
            w = " {} ".format(w)

            altered_corpus = re.sub(w, replacement, altered_corpus)

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

    def run(self, corpus, base_limit=1, token_replacement='UNK'):
        """

        :param corpus:
        :param base_limit:
        :param token_replacement:
        :return:
        """

        # split corpus in tokens
        tokens = self.tokenize_corpus(corpus)

        # count tokens, and create vocabulary-rejection lexicons
        vocabulary_tokens_counts, rejection_tokens_counts = self.create_vocabulary(tokens=tokens,
                                                                                   base_limit=base_limit)

        # replacing uncommon words in original corpus:
        new_corpus = self.replace_uncommon_words(corpus=corpus,
                                                 words=rejection_tokens_counts.keys(),
                                                 replacement=token_replacement)

        # Split new corpus in sentences
        sentences = self.split_to_sentences(corpus=new_corpus)

        # 2. split sentences in tokens
        # 4. Replace uncommon words in main corpus with 'replacement' token

        pass


if __name__ == '__main__':
    a_corpus = "The Cape sparrow (Passer melanurus) is a southern African bird. \n\n " \
               "A medium-sized sparrow at 14–16 centimetres (5.5–6.3 in), it has distinctive grey, brown, and" \
               " chestnut plumage, with large pale head stripes in both sexes. \n\n " \
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

    test_counts = Preprocessor().run(corpus=a_corpus)

    pprint(test_counts)

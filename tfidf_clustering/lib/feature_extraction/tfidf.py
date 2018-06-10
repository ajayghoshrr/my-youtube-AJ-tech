"""
***This module can able to handle the stop words from the directory - stopwords, able to perform \
tf-idf(term frequency inverse document frequency) from the given corpus content in an multidimensional list. \
***tf-idf can also able to perform simple data cleaning like removal of special characters, unicode, and stemming.
***for stemming, made a wrapper from the nltk package.
"""


__author__ = "Author: Ajaighosh Ramachandran"
__date__ = "Date: 2018-06-09 18:30:12 +0530 (Sat, 09 Jun 2018)"
__email__ = "ajayghoshrr@gmail.com"
__status__ = "Development"

# package imports
import os
import re
from collections import Counter
import logging

logging.basicConfig(filename='tfidf.log', level=logging.INFO,
                    format='%(asctime)s:%(filename)s:%(funcName)s:%(levelname)s:%(message)s')


class PreProcessing:
    """
    This class have some pre processing techniques
    """
    def __init__(self):
        self.value = []

    def data_cleaning(self, input_value, sent_tokenize = False):
        """
        Module can able to remove special characters, removing empty lines, change to common case.
            :param input_value: Multidimensional list of strings which is not cleaned
            :type: list
            :param sent_tokenize: after cleaning, returns a common list for complete inout value if sent_tokenize = True\
                else return as in the input_value format
            :type: Boolean
            :return: list of strings
            :type: Iterable
        """

        content = []
        def lower(ch):
            return ch.lower()

        def clean(word):
            return re.sub('[^A-Za-z]+', '', word.strip("\n ? '"))

        if sent_tokenize:
            for i in input_value:
                for j in i:
                    content.append(lower(clean(j)))
        else:
            for i in input_value:
                store_doc = []
                for j in i:
                    store_doc.append(list(map(lower,list(map(clean,j.split())))))
                content.append(store_doc)
        return content

    def remove_stop_words(self,universal_list, stopwords_list=[], sent_removal = False):
        """
        This module can able will remove all the stopwords from the stopwords folder \
        and if stopwords_list contain extra stopwords, that can also removed from universal string

            :param universal_list: list of words which is cleaned
            :type: list
            :param stopwords_list: If any additional stop words needed to be added
            :type: list
            :param sent_removal: If sentence based removal, sent_removal = False, input will be list of sentences else \
                sent_removal = False, input will be list of words
            :type: Iterable
            :return: list of words without stopwords
            :type: Iterable
        """

        root_path = os.path.join(os.path.dirname(__file__),os.pardir, os.pardir, 'datasets', 'stopwords')
        paths = self.collect_paths(root_path)
        list_of_non_preprocess_words = self.verbal_list_converter(paths)
        union_stopwords = self.data_cleaning(list_of_non_preprocess_words + stopwords_list, sent_tokenize= True)
        filtered_words = []
        if sent_removal:
            for sent in universal_list:
                z = []
                for w in sent:
                    if w not in union_stopwords and len(w) != 0:
                        z.append(w)
                filtered_words.append(z)
        else:
            for w in universal_list:
                if w not in union_stopwords:
                    filtered_words.append(w)
        return filtered_words

    def collect_paths(self, root_path):
        """
        Module will collect all the paths of file under root path

            :param root_path: folder path
            :type: str
            :return: collection of paths of files under that directory
            :type: list
        """
        paths = []
        #parsing complete directory for the all possible paths recursively
        for root, dirs, files in os.walk(root_path):
            if len(files) != 0:
                for i in files:
                    paths.append(os.path.join(root, i))
        return paths

    def verbal_list_converter(self, paths):
        """
        This module will convert your document content into the list of strings
            :param paths: list of paths of files for verbal list conversion
            :type: list
            :return: Module will convert the list of files into a multi dimensional array of words
            :type: Iterable
        """

        main_list = []
        for i in paths:
            f = open(i, 'r')
            main_list.append(f.readlines())
            f.close()
        return main_list


class TfIdf(PreProcessing):
    """
    TfIdf package: Yet to be added
    """
    def __init__(self):
        super(TfIdf, self).__init__()


    def tfidf_matrix(self, path_to_corpus, join_corpus = False):
        """
        tfidf_matric can able to convert the corpus content into the tfidf matrix format
            :param path_to_corpus: It's a relative or full path to the corpus folder - Mandatory
            :type: str
            :param join_corpus: If many documents are there and need idf based on each document join_corpus = False
                if idf is based on all the corpus together join_corpus = True - Default parameter
            :type: Boolean
            :return: Iterable
            :type: list
        """

        # calling collect paths, verbal_list_converter, and data cleaning
        content = self.data_cleaning(self.verbal_list_converter(self.collect_paths(path_to_corpus)))
        logging.info("Collected all the paths under corpus path {0}".format(path_to_corpus))
        logging.info("Converted all the files content into the strings")
        logging.info("Cleaned data successfully")
        # removing stopwords from the list
        master = [self.remove_stop_words(i, sent_removal=True) for i in content]
        logging.info("Removed all the stopwords from the stopwords file")
        tfidf = []
        temp = []
        if join_corpus:
            logging.info("Inside the sent_removal = True block")
            for i in master:
                temp += i
                master = [temp]
        
        for doc in master:
            tf_list = []
            # computing the inverse document frequency
            temp = TfIdf.idf_computation(doc)
            logging.info("Computed the idf values")
            # finding the term frequency for complete document
            for tf_value in TfIdf.term_frequency(doc):
                tf_list.append(TfIdf.tf_idf_calculation(tf_value, temp))
            tfidf.append(tf_list)
            logging.info("Calculated the term frequency for the strings")
        return tfidf

    @staticmethod
    def term_frequency(word_list):
        """
        This function will return the term frequency in given list
            :param word_list: word_list
            :type: Iterable
            :return: returns the Dictionary with the Term as key and frequency as value
            :type: Iterable
        """

        tf_dict = []
        for sent in word_list:
            temp_dict = {}
            if len(sent) != 0:
                for words in sent:
                    temp_dict[words] = Counter(sent)[words] / float(len(sent))
                tf_dict.append(temp_dict)
        logging.info("Term frequency successfully found")
        return tf_dict

    @staticmethod
    def idf_computation(universal_list):
        """
        Module returns the idf value of each words in the universal list
            :param universal_list: list of words, It is an universal list or union of words from all the corpus - Mandatory
            :type: Iterable
            :return: idf value of the corpus
            :type: Iterable
        """

        import math
        no_of_document = len(universal_list)
        word_dict = TfIdf.word_dict(universal_list)
        logging.info("Word dict created from the universal list")
        document_count = {}
        for words in word_dict.keys():
            count = 0
            for i in universal_list:
                if words in i:
                    count += 1
            document_count[words] = count
        tf_idf_dict = {}
        # applying the idf formulae
        for words, val in document_count.items():
            tf_idf_dict[words] = math.log(no_of_document/float(document_count[words]))
        logging.info("IDF Formulae applied to all the strings")
        return tf_idf_dict

    @staticmethod
    def tf_idf_calculation(tf_dict, idf_dict):
        """
        Module will calculate the tf_idf for the complete set of words
            :param tf_dict: Dictionary of term frequencies - Mandatory
            :type: Iterable
            :param idf_dict: Dictionary of idf values of the corpus - Mandatory
            :type: Iterable
            :return: tfidf values of corpus key : word, value: tfidf value
            :type: Iterable
        """
        tfidf_dict = {}
        # finding the tf.idf for all the words in the tf dictionary
        for word, val in tf_dict.items():
            tfidf_dict[word] = val * idf_dict[word]
        logging.info("TF-IDF operation successfull")
        return tfidf_dict

    @staticmethod
    def word_dict(content):
        """
        To convert a list into dictioanary with key as word and counts as values
            :param content: List of words
            :type: Iterable
            :return: Returns dictionary of words with count
            :type: Iterable
        """
        word_di = []
        for i in content:
            word_di += i
        logging.info("Inside the word count snippet")
        return dict(Counter(word_di))

# a = TfIdf()
# z = a.tfidf_matrix(os.path.join(os.pardir, os.pardir,'datasets', 'corpus'), join_corpus=False)
# print(z)


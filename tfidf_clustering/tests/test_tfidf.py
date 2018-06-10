"""

This module is intended to test the features of the methods.

"""

import pytest
from lib.feature_extraction import tfidf
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

class TestTfidf:
    main_obj = tfidf.TfIdf()
    tfidf_out = [[{'run': 0.3662040962227032, 'catch': 0.1831020481113516,
                   'sun': 0.06757751801802739, 'sinking': 0.1831020481113516,
                   'racing': 0.1831020481113516},
                  {'sun': 0.05068313851352055, 'relative': 0.13732653608351372,
                   'older': 0.13732653608351372, 'shorter': 0.05068313851352055,
                   'breath': 0.13732653608351372, 'day': 0.13732653608351372,
                   'closer': 0.13732653608351372, 'death': 0.13732653608351372},
                  {'year': 0.10986122886681099, 'shorter': 0.04054651081081644,
                   'find': 0.10986122886681099, 'time': 0.10986122886681099,
                   'plans': 0.10986122886681099, 'naught': 0.10986122886681099,
                   'half': 0.10986122886681099, 'page': 0.10986122886681099,
                   'scribbled': 0.10986122886681099, 'lines': 0.10986122886681099}]]


    def test_data_cleaning_functionality_tc_01(self):
        print(tfidf.TfIdf.data_cleaning.__doc__)
        strings = [["Ajigh9899%$% \n YOuR89?-\n"],["Tyoutube Hello World\n", "Werne1@@123"]]
        obj = tfidf.TfIdf()
        assert (obj.data_cleaning(strings, sent_tokenize=True)) == ['ajighyour', 'tyoutubehelloworld', 'werne']
    def test_tfidf_matrix_functioality_tc_01(self):
        print(tfidf.TfIdf.tfidf_matrix.__doc__)
        path_to_corpus = os.path.join(os.path.dirname(__file__),'data_set')
        obj = tfidf.TfIdf()
        assert (obj.tfidf_matrix(path_to_corpus, join_corpus=False)) == TestTfidf.tfidf_out
    def test_remove_stop_words_function_tc_01(self):
        print(tfidf.TfIdf.remove_stop_words.__doc__)
        obj = tfidf.TfIdf()
        assert (obj.remove_stop_words([['i', 'the', 'we', 'corpus']], sent_removal=True)) == [['corpus']]


    """
    To be added
    """
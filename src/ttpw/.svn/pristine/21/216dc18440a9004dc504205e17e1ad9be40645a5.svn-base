#!/bin/env python
# -*- coding: utf-8 -*-
"""TreeTagger Python wrapper test module.
"""

# See http://pypi.python.org/pypi/unittest2 a backport of most recent unittest
# for older Python versions.
try:
    import unittest2 as unittest
except:
    import unittest
from test import test_support

# Setup parent directory in sys.path.
import sys
from os import path
thedir = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0,thedir)

# Now, import the module to test
import treetaggerwrapper



class TTStartTestCase(unittest.TestCase):
    """Try to start TreeTagger - if this fail..."""

    # Only use setUp() and tearDown() if necessary

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_start_tagger(self):
        pass


class TestCaseNeedingTreeTagger(unittest.TestCase):
    """A test case which need to start TreeTagger, with an ad-hoc language.

    Member tt is an access to the created TreeTagger object.
    """
    def __init__(self,lang) :
        unittest.TestCase.__init__(self)
        self.tt = lang

    def setUp(self):
        self.tt = treetaggerwrapper.TreeTagger(TAGLANG=self.tt)

    def tearDown(self):
        del self.tt # Should close connections.


class TestCase4TextPreprocessing(TestCaseNeedingTreeTagger):
    """Preprocess some texts - must be subclassed with providing."""
    def __init__(self,lang) :
        TestCaseNeedingTreeTagger.__init__(self,lang)
        # sample_and_res is a list of tuples, each tuple has two items, the
        # first one is the string to preprocess, the second one is the attended
        # result (a list of strings). Alls trings must be unicode.
        self.sample_and_res = []

    def runTest(self):
        failedPreprocess = 0
        for sample,res in self.sample_and_res :
            ttres = self.tt.PrepareText(text)
            try :
                self.assertEqual(ttres, res)
            except :
                failedPreprocess += 1
        self.failUnless(failedPreprocess==0)


class EnglishPreprocessingTestCase(TestCase4TextPreprocessing):
    def __init__(self) :
        TestCase4TextPreprocessing.__init__(self,"en")
        self.sample_and_res.append( ( "Hello, Mr Young.",
                                      ["Hello",",","Mr","Young"] ) )



def suite():
    suite = unittest.TestSuite()
    #suite.addTest(WidgetTestCase('test_default_size'))
    #suite.addTest(WidgetTestCase('test_resize'))
    return suite

def test_main():
    suite = unittest.TestLoader().loadTestsFromTestCase(EnglishPreprocessingTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)

    #test_support.run_unittest(
            #TTStartTestCase,
            #EnglishPreprocessingTestCase
            #)

if __name__ == '__main__':
    unittest.main()
    #test_main()

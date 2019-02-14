import doctest

from thesnisright.process import util


def load_tests(loader, tests, ignore):
    del loader, ignore
    doc_test_suite = doctest.DocTestSuite(util, optionflags=doctest.NORMALIZE_WHITESPACE)
    tests.addTests(doc_test_suite)
    return tests

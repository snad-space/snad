import doctest

from thesnisright.interpolate import gp


def load_tests(loader, tests, ignore):
    del loader, ignore
    gp_doc_test_suite = doctest.DocTestSuite(gp, optionflags=doctest.NORMALIZE_WHITESPACE)
    tests.addTests(gp_doc_test_suite)
    return tests

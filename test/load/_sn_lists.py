SNS_NO_CMAIMED_TYPE = frozenset((
    'SNLS-03D3ce',
))

SNS_UPPER_LIMIT = frozenset((
    'SNLS-04D3fq',
    'PS1-10ahf',
    'MLS121209:093512+152855',
))

SNS_E_LOWER_UPPER_MAGNITUDE = frozenset((
    'SNLS-04D3fq',
))

SNE_E_TIME = frozenset((
    'MLS121209:093512+152855',
))

SNS_UNORDERED_PHOTOMETRY = frozenset((
    'PTF09atu',
    'PS1-10ahf',
))

SNS_HAVE_ZERO_E_MAGNITUDE = frozenset((
    'Gaia14ado',
))

SNS_HAVE_NOT_PHOTOMETRY = frozenset((
    'GRB 081025A',
))

SNS_HAVE_NOT_MAGN_ERRORS = frozenset((
    'SN2005V',
))

SNS_ZERO_VALID_PHOTOMETRY_DOTS = frozenset((
    'SN2007bk',
))

SNS_HAVE_SPECTRA = frozenset((
    'SNLS-04D3fq',
))

SNS_HAVE_NOT_SPECTRA = frozenset((
    'Gaia14ado',
))

SNS_HAVE_B_BAND = frozenset((
    'SN1993A',
))

SNS_ALL = frozenset.union(SNS_NO_CMAIMED_TYPE, SNS_UPPER_LIMIT, SNS_E_LOWER_UPPER_MAGNITUDE, SNE_E_TIME,
                          SNS_UNORDERED_PHOTOMETRY, SNS_HAVE_ZERO_E_MAGNITUDE, SNS_HAVE_B_BAND)
SNS_ALL_TUPLE = tuple(sorted(SNS_ALL))

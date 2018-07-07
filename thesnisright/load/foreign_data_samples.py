from pandas import read_csv


def load_pantheon_sample(filename):
    """Load Pantheon sample from .dat file

    Pantheon is a cosmological SNIa sample used in D. M. Scolnic, et al. 2018.
    This data file can be located at data/spec_class_samples/pantheon_sne.dat.
    The origin of this file is located on:
    https://github.com/dscolnic/Pantheon/blob/master/lcparam_full_long.txt
    See GitHUb page of the project for details:
    https://github.com/dscolnic/Pantheon
    http://adsabs.harvard.edu/abs/2018ApJ...859..101S

    Parameters
    ----------
    filename: str

    Returns
    -------
    panads.DataFrame
    """
    data = read_csv(filename, sep=' ')
    return data


def get_pantheon_sne_suffixes(filename):
    """Get Pantheon SN name suffixes from .dat file

    Pantheon is a cosmological SNIa sample used in D. M. Scolnic, et al. 2018.
    This data file can be located at data/spec_class_samples/pantheon_sne.dat.
    The origin of this file is located on:
    https://github.com/dscolnic/Pantheon/blob/master/lcparam_full_long.txt
    See GitHUb page of the project for details:
    https://github.com/dscolnic/Pantheon
    http://adsabs.harvard.edu/abs/2018ApJ...859..101S

    Parameters
    ----------
    filename: str

    Returns
    -------
    list
    """
    sample = load_pantheon_sample(filename)
    return list(sample['#name'])


def load_anderson_sample(filename):
    """Load Anderson, et al. 2014 SN II data from csv file

    These data can be obtained from Anderson, et al. II, table 2.
    This data file can be located at data/spec_class_samples/anderson_sne.csv
    http://adsabs.harvard.edu/abs/2014ApJ...786...67A

    Parameters
    ----------
    filename: str

    Returns
    -------
    panads.DataFrame
    """
    data = read_csv(filename, sep=',')
    return data


def get_anderson_sne(filename):
    """Get Sanders SN names, et al. 2015 SN IIP data from tsv file

    These data can be obtained from Anderson, et al. II, table 2.
    This data file can be located at data/spec_class_samples/anderson_sne.csv
    http://adsabs.harvard.edu/abs/2014ApJ...786...67A

    Parameters
    ----------
    filename: str

    Returns
    -------
    list
    """
    sample = load_anderson_sample(filename)
    return ['SN{}'.format(name) for name in sample.SN]



def load_sanders_sample(filename):
    """Load Sanders, et al. 2015 SN IIP data from tsv file

    Sanders, et al. IIP data can be downloaded from Vizier in TSV format.
    This data file can be located at data/spec_class_samples/sanders_sne.tsv
    http://adsabs.harvard.edu/abs/2015ApJ...799..208S

    Parameters
    ----------
    filename: str

    Returns
    -------
    panads.DataFrame
    """
    data = read_csv(filename, sep=';', comment='#')
    return data


def get_sanders_sne(filename):
    """Get Sanders SN names, et al. 2015 SN IIP data from tsv file

    In Sanders, et al. IIP data can be downloaded from Vizier in TSV format.
    This data file can be located at data/spec_class_samples/sanders_sne.tsv
    http://adsabs.harvard.edu/abs/2015ApJ...799..208S

    Parameters
    ----------
    filename: str

    Returns
    -------
    list
    """
    sample = load_sanders_sample(filename)
    names = ['PS1'+name.strip() for name in sample.PS1 if name.startswith('-1')]
    return names


def load_cccp_sample(filename):
    """Load CCCP data from csv file

    CCCP (Caltech core-collapsed Program) data can be located at
    data/spec_class_samples/cccp_sne.csv of original repo. Please cite related
    papers, look program page for more details:
    http://www.astro.caltech.edu/~avishay/cccp.html

    Parameters
    ----------
    filename: str

    Returns
    -------
    panads.DataFrame
    """
    data = read_csv(filename, sep=',')
    return data


def get_cccp_sne(filename):
    """Get CCCP SN names from csv file

    CCCP (Caltech core-collapsed Program) data can be located at
    data/spec_class_samples/cccp_sne.csv of original repo. Please cite related
    papers, look program page for more details:
    http://www.astro.caltech.edu/~avishay/cccp.html

    Parameters
    ----------
    filename: str

    Returns
    -------
    list
    """
    sample = load_cccp_sample(filename)
    return list(sample.Name)

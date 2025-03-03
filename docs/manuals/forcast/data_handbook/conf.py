# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Astropy documentation build configuration file.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this file.
#
# All configuration values have a default. Some values are defined in
# the global Astropy configuration which is loaded here before anything else.
# See astropy.sphinx.conf for which values are set there.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('..'))
# IMPORTANT: the above commented section was generated by sphinx-quickstart, but
# is *NOT* appropriate for astropy or Astropy affiliated packages. It is left
# commented out with this explanation to make it clear why this should not be
# done. If the sys.path entry above is added, when the astropy.sphinx.conf
# import occurs, it will import the *source* version of astropy instead of the
# version installed (if invoked as "make html" or directly with sphinx), or the
# version in the build directory (if "python setup.py build_sphinx" is used).
# Thus, any C-extensions that are needed to build the documentation will *not*
# be accessible, and the documentation will not build correctly.

import os
import sys
import datetime

try:
    from sphinx_astropy.conf.v1 import *  # noqa
except ImportError:
    print('ERROR: the documentation requires the sphinx-astropy package to be installed')
    sys.exit(1)

# Get configuration information from setup.cfg
from configparser import ConfigParser
conf = ConfigParser()

conf.read([os.path.join(os.path.dirname(__file__),
                        '..', '..', '..', '..', 'setup.cfg')])
setup_cfg = dict(conf.items('metadata'))

# -- General configuration ----------------------------------------------------

# By default, don't highlight syntax in literals
highlight_language = 'none'

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.2'

# To perform a Sphinx version check that needs to be more specific than
# major.minor, call `check_sphinx_version("x.y.z")` here.
# check_sphinx_version("1.2.1")

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns.append('_templates')

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.
rst_epilog += """
"""

# -- Project information ------------------------------------------------------

# NASA document number and revision
docnumber = 'FORCAST Data Handbook'
docrev = 'M'

# This does not *have* to match the package name, but typically does
project = setup_cfg['name']
author = setup_cfg['author']
copyright = '{0}, {1}'.format(
    datetime.datetime.now().year, setup_cfg['author'])

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

package = docnumber
version = 'Rev. %s' % docrev
release = ': %s %s' % (package, version)


# -- Options for LaTeX output -------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto/manual]).
latex_documents = [('data_handbook',
                    'forcast_data_handbook.tex',
                    "Guest Observer Handbook for FORCAST Data Products ",
                    "M. Clarke, W. Vacca, E. Chambers, J. Radomski",
                    'howto')
                   ]

# Fix environment error, make one-sided document,
# add NASA header and footer
latex_elements = {
    'classoptions': ',openany,oneside',
    'babel': r'\usepackage[english]{babel}',
    'maxlistdepth': 20,
    'printindex': r'\footnotesize\raggedright\printindex',
    'preamble': r'''
\pagestyle{plain}
\setcounter{tocdepth}{2}
\usepackage{fancyhdr}
\fancypagestyle{normal}{
\fancyhf{}
\fancyhead[R]{%s\\Rev. %s}
\fancyfoot[C]{\textbf{VERIFY THAT THIS IS THE CORRECT REVISION BEFORE USE}\\\thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}}
''' % (docnumber, docrev),
}

# number figures for manuals
numfig = True

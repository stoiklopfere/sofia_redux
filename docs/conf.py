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

import sys
import datetime
from importlib import import_module

try:
    from sphinx_astropy.conf.v1 import *  # noqa
except ImportError:
    print('ERROR: the documentation requires the sphinx-astropy package to be installed')
    sys.exit(1)

# -- General configuration ----------------------------------------------------

# By default, don't highlight syntax in literals
highlight_language = 'none'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns.append('_templates')

# exclude some things intended for direct inclusion or for latex/html
# specific support
exclude_patterns.append('sofia_redux/pipeline/redux_usage.rst')
exclude_patterns.append('sofia_redux/pipeline/usage/*.rst')
exclude_patterns.append('manuals/*/*/index.rst')
exclude_patterns.append('manuals/*/*/redux_doc.rst')
exclude_patterns.append('manuals/*/users/data_description.rst')
exclude_patterns.append('manuals/*/users/software_description.rst')
exclude_patterns.append('manuals/*/users/spectral_extraction.rst')
exclude_patterns.append('manuals/*/users/spectral_calibration.rst')
exclude_patterns.append('manuals/*/users/spectral_display.rst')
exclude_patterns.append('manuals/*/data_handbook')
exclude_patterns.append('manuals/*/*/api')

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.
rst_epilog += """
"""

# -- Project information ------------------------------------------------------

# This does not *have* to match the package name, but typically does
project = "sofia_redux"
author ="SOFIA-USRA"
copyright = '{0}, {1}'.format(
    datetime.datetime.now().year, author)

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

import_module(project)
package = sys.modules[project]

# The short X.Y version.
version = package.__version__.split('-', 1)[0]
# The full version, including alpha/beta/rc tags.
release = package.__version__


# -- Options for HTML output --------------------------------------------------

# A NOTE ON HTML THEMES
# The global astropy configuration uses a custom theme, 'bootstrap-astropy',
# which is installed along with astropy. A different theme can be used or
# the options for this theme can be modified by overriding some of the
# variables set in the global configuration. The variables set in the
# global configuration are listed below, commented out.


# Add any paths that contain custom themes here, relative to this directory.
# To use a different custom theme, add the directory containing the theme.
html_theme_path = ['']

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes. To override the custom theme, set this to the
# name of a builtin theme or the name of a custom theme in html_theme_path.
html_theme = 'bootstrap-sofia'


# Please update these texts to match the name of your package.
html_theme_options = {
    'logotext1': 'SOFIA',  # white,  semi-bold
    'logotext2': 'Redux',  # orange, light
    'logotext3': ':docs'   # white,  light
    }


# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = ''

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = os.path.join('_static', 'redux.ico')

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = ''

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = '{0} v{1}'.format(project, release)

# Output file base name for HTML help builder.
htmlhelp_basename = project + 'doc'


# -- Options for LaTeX output -------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [('index', project + '.tex', 'SOFIA Redux Documentation',
                    author, 'howto')]

# Fix environment error, make one-sided document,
# allow deeply nested lists
latex_elements = {
    'classoptions': ',openany,oneside',
    'babel': r'\usepackage[english]{babel}',
    'maxlistdepth': 20,
    'printindex': r'\footnotesize\raggedright\printindex',
    'preamble': r'''
\pagestyle{plain}
\setcounter{tocdepth}{2}
'''
}

# number figures for manuals
numfig = True

# -- Options for manual page output -------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [('index', project.lower(), project + u' Documentation',
              [author], 1)]

# -- Customize inheritance diagram --------------------------------------------
# make the labels fit in the nodes
inheritance_node_attrs = dict(margin=0.25)
# UML style inheritance arrows
inheritance_graph_attrs = dict(rankdir="LR", size='""')
inheritance_edge_attrs = dict(arrowtail="empty", arrowsize=1.2, dir="back")

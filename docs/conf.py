# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most generic options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'PyADTS'
copyright = '2020, Xiao Qinfeng'
author = 'Xiao Qinfeng'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
    'sphinx.ext.napoleon',
    'nbsphinx'
]

bibtex_bibfiles = ['ref.bib']
latex_engine = 'xelatex'
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'preamble': r'''
\usepackage{fontspec}
\defaultfontfeatures{Mapping=tex-text,Scale=MatchLowercase}
\setmainfont{Times}
\setmonofont{Lucida Sans Typewriter}
    '''
}

autosummary_generate = True  # = ["index"] to skip API

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'

# import guzzle_sphinx_theme
#
# # Adds an HTML table visitor to apply Bootstrap table classes
# html_translator_class = 'guzzle_sphinx_theme.HTMLTranslator'
# html_theme_path = guzzle_sphinx_theme.html_theme_path()
# html_theme = 'guzzle_sphinx_theme'
#
# # Register the theme as an extension to generate a sitemap.xml
# extensions.append("guzzle_sphinx_theme")
#
# # Guzzle theme options (see theme.conf for more information)
# html_theme_options = {
#     # Set the name of the project to appear in the sidebar
#     "project_nav_name": "PyADTS",
#     "base_url": "",
# }

html_theme = 'furo'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "github_url": "https://github.com/larryshaw0079/PyADT",
    # "show_prev_next": False,
    # "external_links": [
    #     {"name": "Webpage", "url": "https://scikit-multiflow.github.io/"},
    # ],
    # "use_edit_page_button": False,
}

# import sphinx_minoo_theme
#
# html_theme = "sphinx_minoo_theme"
#
# html_theme_path = [sphinx_minoo_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Other Custom Options -------------------------------------------------

# from recommonmark.parser import CommonMarkParser
# source_parsers = {
#     '.md': CommonMarkParser,
# }

source_suffix = ['.rst']

master_doc = 'index'

autodoc_typehints = "description"
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

suppress_warnings = [
    'nbsphinx',
]

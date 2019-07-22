# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'xenith'
copyright = '2019, William E. Fondrie'
author = 'William E. Fondrie'

#import xenith
#version = xenith.__version__
#release = xenith.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

autosummary_generate = True
napoleon_use_ivar = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Random things --------------------------------------------------------------
source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'
htmlhelp_basename = "xenith_doc"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
html_theme_options = {
    "analytics_id": "UA-78640578-2",
    "logo_name": False,
    "logo": "xenith_logo.svg",
    "description": ("Enhanced cross-linked peptide detection using pretrained"
                    " models."),
    "github_user": "wfondrie",
    "github_repo": "xenith",
    "codecov_button": True,
    "github_button": True,
    "travis_button": True,
    "show_powered_by": False,
    "caption_font_size": "14pt",
    "caption_font_family": "Garamond"
}

html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

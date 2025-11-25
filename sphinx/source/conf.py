# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pLAST'
copyright = '2025, Kamil Krakowski, Stanisław Dunin-Horkawicz'
author = 'Kamil Krakowski, Stanisław Dunin-Horkawicz'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "sphinx.ext.mathjax",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "repository_url": "https://github.com/labstructbioinf/pLAST",
    "use_repository_button": True,
}

html_context = {
    "display_github": True,
    "github_user": "labstructbioinf",
    "github_repo": "pLAST",
    "github_version": "main",
    "conf_py_path": "/sphinx/source/",
}

html_static_path = ['_static']

nbsphinx_prolog = r"""
.. raw:: html

    <div class="admonition note">
        <p><a href="{{ env.doc2path(env.docname, base=None) }}" download>
        📥 Download notebook (.ipynb)
        </a></p>
    </div>
"""


import os
import sys
sys.path.insert(0, os.path.abspath("../../"))
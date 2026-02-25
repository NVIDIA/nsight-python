# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "nsight-python"
copyright = "2025, NVIDIA"
author = "NVIDIA"
release = importlib.metadata.version("nsight-python")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx_autodoc_typehints",  # Show type hints
    "sphinx_mdinclude",  # Include markdown in rst docs
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# -- Intersphinx configuration ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nvidia_sphinx_theme"
html_static_path = ["_static"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "fullname": True,
}

autodoc_typehints = "description"
typehints_use_rtype = True
typehints_use_signature = True

toc_object_entries_show_parents = "all"

html_theme_options = {
    "switcher": {
        "json_url": "https://docs.nvidia.com/nsight-python/_static/switcher.json",
        "version_match": release,
    }
}

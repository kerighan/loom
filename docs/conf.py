"""Sphinx configuration for the loom documentation."""

import os
import sys

# Make the loom package importable for autodoc.
sys.path.insert(0, os.path.abspath(".."))

project = "loom"
author = "kerighan"
copyright = "2026, kerighan"
release = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",       # Google / NumPy style docstrings
    "sphinx.ext.viewcode",       # [source] links
    "sphinx.ext.intersphinx",
    "myst_parser",               # Markdown (.md) source files
]

# Markdown + reStructuredText both accepted.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

myst_enable_extensions = ["deflist", "colon_fence", "fieldlist", "tasklist"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ── HTML output ───────────────────────────────────────────────────────────
html_theme = "furo"
html_title = "loom"
html_static_path = ["_static"] if os.path.isdir(os.path.join(os.path.dirname(__file__), "_static")) else []
html_theme_options = {
    "sidebar_hide_name": False,
}

# ── autodoc ─────────────────────────────────────────────────────────────────
autodoc_member_order = "bysource"
autodoc_typehints = "description"
# No "members: True" default — each directive in api.rst lists exactly the
# members to document, so internal helpers aren't pulled in.
autodoc_default_options = {
    "show-inheritance": True,
}
# Pydantic / numpy may be heavy or optional at doc-build time — mock if missing.
autodoc_mock_imports = []

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_admonition_for_examples = True   # render "Example:" blocks cleanly

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

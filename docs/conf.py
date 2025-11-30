import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

project = "Bias Amplification"
copyright = "2025, Rahul, Ushnesha and Bhanu"
author = "Rahul Nair, Ushnesha Daripa and Bhanu Tokas"
release = "0.1.1"

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_title = "Bias Amplification"
html_short_title = "Bias Amplification"
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True
html_show_sphinx = False

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
}

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("<https://docs.python.org/3>", None),
    "numpy": ("<https://numpy.org/doc/stable/>", None),
    "pandas": ("<https://pandas.pydata.org/docs/>", None),
    "torch": ("<https://pytorch.org/docs/stable/>", None),
    "sklearn": ("<https://scikit-learn.org/stable/>", None),
}

# NumPy docstring settings
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = True

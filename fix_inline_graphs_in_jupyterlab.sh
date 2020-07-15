#!/bin/bash
conda install jupyterlab=1.0 "ipywidgets=7.5"

# Avoid "JavaScript heap out of memory" errors during extension installation
# (OS X/Linux)
export NODE_OPTIONS=--max-old-space-size=4096
# (Windows)
#set NODE_OPTIONS=--max-old-space-size=4096

# Jupyter widgets extension
jupyter labextension install @jupyter-widgets/jupyterlab-manager@1.0 --no-build

# FigureWidget support
jupyter labextension install plotlywidget@1.0.0 --no-build

# and jupyterlab renderer support
jupyter labextension install jupyterlab-plotly@1.0.0 --no-build

# JupyterLab chart editor support (optional)
jupyter labextension install jupyterlab-chart-editor@1.2 --no-build

# Build extensions (must be done to activate extensions since --no-build is used above)
jupyter lab build

# Unset NODE_OPTIONS environment variable
# (OS X/Linux)
unset NODE_OPTIONS
# (Windows)
#set NODE_OPTIONS=

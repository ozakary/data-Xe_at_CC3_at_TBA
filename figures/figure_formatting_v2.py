import scipy.interpolate as interpolate
import matplotlib
import matplotlib.image as image
from matplotlib import rc, rcParams
import numpy as np

# Global formatting options

nearly_black = '#161616'
light_grey = '#EEEEEE'
lighter_grey = '#F5F5F5'
white = '#FFFFFF'
light_blue = '#6d9fd1'

# dark blue, teal, gold, orange, light orange
#colors = ['#264653', '#2A9D8F', '#E9C46A', '#E76F51', '#F3B6A5']

fontsize_1 = 20
fontsize_2 = 18
fontsize_3 = 18
fontsize_4 = 16
fontsize_5 = 16

tableau10 = [ '#4379a7', '#f28e2b', '#e15759', 
              '#76b7b2', '#59a14f', '#edc948',
              '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac' ]

master_formatting = {
    'xtick.major.pad': 12,
    'ytick.major.pad': 12,
    'ytick.color': nearly_black,
    'xtick.color': nearly_black,
    'xtick.direction': 'out',
    'xtick.major.size': 8,
    'ytick.direction': 'out',
    'ytick.right': False,
    'ytick.major.size': 8,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'axes.labelcolor': nearly_black,
    'axes.labelsize': fontsize_2,
    'legend.facecolor': light_grey,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'mathtext.fontset': 'custom',
    'font.size': fontsize_1,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Helvetica Neue',
    'mathtext.it': 'Helvetica Neue:italic',
    'mathtext.bf': 'Helvetica Neue:bold',
    'savefig.bbox': 'tight',
    'axes.facecolor': white,
    'axes.labelpad': 10.0,
    'axes.titlesize': fontsize_3,
    'axes.titlepad': 32,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 1.5,
    'lines.markersize': 10.0,
    'lines.markeredgewidth': 1.5,
    'lines.linewidth': 1.5,
    'lines.scale_dashes': False,
    'xtick.labelsize': fontsize_4,
    'ytick.labelsize': fontsize_4,
    'legend.fontsize': fontsize_5,

    'text.latex.preamble': r"\usepackage{amssymb}\n\usepackage{amsmath",
}

# Function to set rcParams using the formatting dictionary
def set_rcParams(formatting):
    for k, v in formatting.items():
        if k == 'text.latex.preamble' and isinstance(v, list):
            v = "\n".join(v)
        rcParams[k] = v

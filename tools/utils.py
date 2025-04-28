import torch
import pandas as pd
import numpy as np

import cellrank as cr
from anndata import AnnData
from scvelo import logging as logg
import os,shutil
from typing import Dict, Optional, Sequence, Tuple, Union

from .._model import REGVELOVI

def split_elements(character_list):
    """split elements."""
    result_list = []
    for element in character_list:
        if '_' in element:
            parts = element.split('_')
            result_list.append(parts)
        else:
            result_list.append([element])
    return result_list

def combine_elements(split_list):
    """combine elements."""
    result_list = []
    for parts in split_list:
        combined_element = "_".join(parts)
        result_list.append(combined_element)
    return result_list

def get_list_name(lst):
    names = []
    for name, obj in lst.items():
        names.append(name)
    return names

def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


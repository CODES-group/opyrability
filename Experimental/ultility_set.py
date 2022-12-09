# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 14:54:31 2022

@author: San Dinh
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def makeplot(AOS_poly):
    fig, ax = plt.subplots()
    for i in range(len(AOS_poly)):
        x = AOS_poly[i][0]
        y = AOS_poly[i][1]
        order = np.argsort(np.arctan2(y - y.mean(), x - x.mean()))
        A = AOS_poly[i].T
        polygon = Polygon(A[order] ,facecolor='tab:blue', edgecolor='black', linewidth=1)
        ax.add_patch(polygon)
    ax.autoscale_view()
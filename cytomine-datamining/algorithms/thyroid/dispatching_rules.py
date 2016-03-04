# -*- coding: utf-8 -*-

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"

import numpy as np
from sldc import DispatchingRule


class AggregateRule(DispatchingRule):

    def __init__(self, cell_max_area, cluster_min_cell_nb):
        """Constructor for AggregateRule object

        Constructor
        -----------
        cell_max_area : float
            The cells maximum area. It must be consistent with the polygon
            coordinate system. In particular with the scale
        cluster_min_cell_nb : int
            The minimum number of cells to form a cluster. It must be consistent
            with the polygon coordinate system. In particular with the scale
        """
        self._cell_max_area = cell_max_area
        self._cluster_min_cell_nb = cluster_min_cell_nb

    def evaluate(self, polygon):
        return polygon.area >= self._cluster_min_cell_nb * self._cell_max_area


class SmallClusterRule(DispatchingRule):

    def __init__(self, cell_min_area, cell_max_area, cluster_min_cell_nb):
        """Constructor for SmallClusterRule object

        Constructor
        -----------
        cell_min_area : float
            The cells minimum area. It must be consistent with the polygon
            coordinate system. In particular with the scale
        cell_max_area : float
            The cells maximum area. It must be consistent with the polygon
            coordinate system. In particular with the scale
        cluster_min_cell_nb : int
            The minimum number of cells to form a cluster. It must be consistent
            with the polygon coordinate system. In particular with the scale
        """
        self._cell_max_area = cell_max_area
        self._cell_min_area = cell_min_area
        self._cluster_min_cell_nb = cluster_min_cell_nb

    def evaluate(self, polygon):
        return self._cell_max_area < polygon.area < self._cluster_min_cell_nb * self._cell_max_area


class CellRule(DispatchingRule):

    def __init__(self, cell_min_area, cell_max_area, cell_min_circularity, cluster_min_cell_nb):
        """Constructor for CellRule object

        Constructor parameters
        ----------------------
        cell_min_area : float
            The cells minimum area. It must be consistent with the polygon
            coordinate system. In particular with the scale
        cell_max_area : float
            The cells maximum area. It must be consistent with the polygon
            coordinate system. In particular with the scale
        cell_min_circularity : float
            The cells minimum circularity. It must be consistent with the polygon
            coordinate system. In particular with the scale
        cluster_min_cell_nb : int
            The minimum number of cells to form a cluster. It must be consistent
            with the polygon coordinate system. In particular with the scale
        """
        self._cell_max_area = cell_max_area
        self._cell_min_area = cell_min_area
        self._cell_min_circularity = cell_min_circularity
        self._cluster_min_cell_nb = cluster_min_cell_nb

    def evaluate(self, polygon):
        circularity = 4 * np.pi * polygon.area / (polygon.length * polygon.length)
        return self._cell_min_area <= polygon.area <= self._cell_max_area and circularity > self._cell_min_circularity
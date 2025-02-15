import numpy as np
import logging

from path import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class OrderParameter:
    def __init__(self, dim=1):
        """Initialize the Ensemble object.mro

        Parameters
        ----------
        params : dict
            Dictionary of parameters for the order parameter.

        """
        self.dim = dim

    def calculate(self, ph):
        """ Calculate the order parameter for a phasepoint.

        Parameters
        ----------
        ph : tuple
            The phasepoint for which the order parameter is calculated.

        Returns
        -------
        order : list of floats
            The order parameter for the phasepoint.
        """
        # Here, we just return the x coordinate of the phasepoint
        if self.dim == 1:
            return [ph[0]]
        elif self.dim == 2:
            return [ph[0][1], ph[0][0]]
        else:
            raise ValueError(f"Order parameter dim {self.dim} not supported.")
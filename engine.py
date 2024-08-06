import numpy as np
import logging
from potential import Potential
from potential import Maze2D_color, RectangularGridWithBarrierPotential, RectangularGridWithRuggedPotential

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class LangevinEngine:
    """ Representation of a Langevin engine in PPTIS

    Attributes
    ----------
    settings : dict
        Dictionary containing the settings of the engine
    dt : float
        Time step of the engine
    temperature : float
        Temperature of the engine
    friction : float
        Friction coefficient of the engine
    dim : int
        Dimension of the engine
    potential : :py:class:`Potential` object
        Potential of the engine
    phasepoint : tuple (x, v) of floats
        Phasepoint of the engine

    """
    def __init__(self, settings):
        self.settings = settings
        self.dt = self.settings["dt"]
        self.temperature = self.settings["temperature"]
        self.friction = self.settings["friction"]
        self.potential = Potential()
        self.phasepoint = None

    def step(self, ph=None):
        """Performs a single step of the Langevin engine.

        Parameters
        ----------
        ph : tuple (x, v) of floats
            Phasepoint at which to perform the step

        Returns
        -------
        ph : tuple (x, v) of floats
            Phasepoint after the step
        """
        if ph is None:
            ph = (self.phasepoint[0], self.phasepoint[1])
        x, v = ph
        # Calculate the potential and force at the current phasepoint
        _, force = self.potential.potential_and_force(ph)

        # Calculate the stochastic force based on the temperature and friction
        stochastic_force = \
            np.sqrt(2 * self.temperature * self.friction) * np.random.randn()

        # Update the velocity and position using the Langevin equation
        v = v + (force - self.friction * v) * self.dt +\
              stochastic_force * np.sqrt(self.dt)
        x = x + v * self.dt
        self.phasepoint = (x,v)
        return self.phasepoint

    def draw_velocities(self):
        """Draws velocities from the Maxwell-Boltzmann distribution.

        Returns
        -------
        v : float
            Velocity drawn from the Maxwell-Boltzmann distribution
        """
        return np.sqrt(self.temperature) * np.random.randn()

    def set_phasepoint(self, ph):
        """ Sets the phasepoint of the engine.

        Parameters
        ----------
        ph : tuple (x, v) of floats
            Phasepoint to set the engine to

        """
        self.phasepoint = ph

class ndLangevinEngine:
    def __init__(self, settings):
        """
        Initialize LangevinEngine.

        Parameters:
        - dt: timestep (float)
        - T: temperature (float)
        - gamma: friction coefficient (float)
        - dim: dimension of the engine (int, e.g., 1D, 2D, 3D, ...)
        - potential: Potential class with 'force' function
        """
        self.settings = settings
        self.dt = self.settings["dt"]
        self.T = self.settings["temperature"]
        self.gamma = self.settings["friction"]
        #self.potential = Maze2D_color(mazefig="maze.png")
        self.potential = RectangularGridWithBarrierPotential(3*0.1, 9*0.1, 2000, (1.5*0.1, 4.5*0.1), -3.5*3/10, 0.2, 2*.1)
        #self.potential = RectangularGridWithRuggedPotential()
        self.phasepoint = None
        self.mass = self.settings["mass"]
        self.dim = self.settings["dim"]
        self.kB = 1.0
        self.kT = self.kB * self.T
        self.beta = 1.0 / self.kT
        self.sigma = np.sqrt(2.0 * self.kT * self.gamma * self.dt / self.mass)
        self.gammadt = self.gamma * self.dt
        self.dtdivmass = self.dt / self.mass
        self.one_minus_gammadt = 1.0 - self.gammadt
        self.equipartition_sigma = np.sqrt(self.kT / self.mass)
        self.mdsteps = 0
    
    def step(self, ph):
        """
        Perform a single integration step starting from phasepoint ph = (x, v).

        Parameters:
        - ph: phasepoint tuple (x, v)

        Returns:
        - new_ph: new phasepoint tuple (x_new, v_new)
        """

        x, v = ph
        # Calculate the new position using the current velocity.
        x_new = x + v * self.dt
        pot, force = self.potential.potential_and_force((x_new, v))

        
       # calculate the stochastic force
        xi = np.random.normal(0, 1, self.dim)

        # update the velocity and position using the Langevin equation
        v_new = v * self.one_minus_gammadt\
                + (force * self.dtdivmass)\
                + self.sigma * xi
        
        self.mdsteps += 1

        return (x_new, v_new)
    
    def draw_velocities(self):
        """
        Draw velocities from the Maxwell-Boltzmann distribution at temperature T
        for a particle of dimension dim. 

        Returns:
        v : np.array
            Velocity drawn from the Maxwell-Boltzmann distribution
        """
        # Maxwell-Boltzmann distribution for each component of the velocity
        return np.random.normal(0, self.equipartition_sigma, self.dim)

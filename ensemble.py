import logging
import numpy as np
from path import Path
from filehandler import make_ens_dirs_and_files
from engine import LangevinEngine, ndLangevinEngine
from order import OrderParameter
import pickle as pkl
from funcs import remove_lines_from_file

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PATH_FMT = (
    '{0:>10d} {1:>10d} {2:>10d} {3:1s} {4:1s} {5:1s} {6:>7d} '
    '{7:3s} {8:2s} {9:>16.9e} {10:>16.9e} {11:>7d} {12:>7d} '
    '{13:>16.9e} {14:>7d} {15:7d} {16:>16.9e}'
)

class Ensemble:
    """ Representation of an ensemble in PPTIS

    Attributes
    ----------

    id : int
        Unique identifier of the ensemble
    intfs : dict of floats
        Dictionary of the L M R interfaces of the ensemble
    paths : list of :py:class:`Path` objects
        List of previously accepted paths in the ensemble
    data : list
        List containing data on the paths generated for each cycle in the
        ensemble
    max_len : int
        Maximum length of a path in the ensemble
    settings : dict
        Dictionary containing the settings of the ensemble
    start_conditions : set of str
        Set containing the start conditions of the ensemble
    end_conditions : set of str
        Set containing the end conditions of the ensemble
    cross_conditions : set of str
        Set containing the cross conditions of the ensemble
    ens_type : str
        String indicating the type of ensemble
    engine : :py:class:`LangevinEngine` object
        Engine of the ensemble
    paths : list of :py:class:`Path` objects
        List of paths in the ensemble
    max_paths : int
        Maximum number of paths that are kept in memory for the ensemble
    cycle : int
        Cycle number of the ensemble, i.e. the number of paths that have been
        generated for the ensemble (including rejected paths).
    acc_cycle : int
        How many cycles resulted in an accepted path?
    md_cycle : int
        How many cycles included the MD engine to generate the accepted paths?
    last_path : :py:class:`Path` object
        Last path that was **accepted** for the ensemble
    name : str
        Name of the ensemble
    order : :py:class:`OrderParameter` object
        Order parameter of the ensemble
    extremal_conditions : set of str
        Set containing the extremal conditions of the ensemble. This is the
        union of the start and end conditions.
    simtype : str
        Type of simulation to perform. This is either "repptis" or "retis"
    """

    def __init__(self, settings):
        """Initialize the Ensemble object.

        Parameters
        ----------
        settings : dict
            Dictionary containing the settings of the simulation
        """

        self.id = settings["id"]
        make_ens_dirs_and_files(self.id)
        # for infswap, we need a separate pe file
        self.infpe = f"pathensemble{self.id[0]}.txt"
        with open(self.infpe, "w+") as f: 
            f.write("")
        self.settings = settings
        self.intfs = self.settings["intfs"]
        self.intfvals = list(self.intfs.values())
        self.max_len = self.settings["max_len"]
        self.ens_type = self.settings["ens_type"]
        self.paths = []
        self.data = []
        self.max_paths = self.settings["max_paths"]
        self.name = settings.get("name", str(id))
        self.cycle = 0
        self.cycle_acc = 0
        self.cycle_md = 0
        self.simtype = settings["simtype"]
        self.prime_both_starts = settings.get("prime_both_starts", False)
        self.level = settings.get("level", 0)
        self.dim = settings.get("dim", 1)

        # Set the start, end and cross conditions of the ensemble
        self.set_conditions()
        # extremal conditions is the union of start and end conditions
        self.extremal_conditions = self.start_conditions.union(
            self.end_conditions)

        # Set the engine of the ensemble
        self.set_engine()

        # Set the order parameter of the ensemble
        self.set_order_parameter()

    def set_engine(self):
        """ Sets the engine of the ensemble.

        """
        #self.engine = LangevinEngine(self.settings)
        self.engine = ndLangevinEngine(self.settings)

    def set_order_parameter(self):
        self.orderparameter = OrderParameter(self.dim)

    def update_data(self, status, trial, gen, simcycle,
                    update_connectivity=False):
        """Updates the data of the path ensemble after a move has been
        performed. If the path is accepted, the last_path and paths attributes
        are updated.

        Parameters
        ----------
        status : str
            Status of the move. This is either "ACC" or any of the "REJ" flags.
        trial : :py:class:`Path` object
            Trial path
        gen : int
            Generation of the move: swap (s-, s+), shoot (sh), null (00)
        simcycle : int
            Cycle number of the simulation
        update_paths : bool, optional
            Whether to update the last_path and paths attributes of the
            ensemble. Some moves do this management themselves. 
            Default is True.

        """
        # path data to obtain:
        trial.omin = min([op[0] for op in trial.orders])
        trial.omax = max([op[0] for op in trial.orders])
        trial.ptype = self.get_ptype(trial)
        trial.plen = len(trial.phasepoints)
        trial.simcycle = simcycle
        trial.status = status
        trial.gen = gen
        if gen == "ex":
            trial.cycle_acc = 0
            trial.cycle_md = 0
            return
        self.cycle += 1
        if self.simtype in ["retis", "repptis", "forced"]:
            if gen == "sh":
                self.cycle_md += 1
        if self.simtype in ["repptis", "forced"]:
            if gen in ["s-", "s+"]:
                self.cycle_md += 1
        if status == "ACC":
            self.cycle_acc += 1
            self.last_path = trial
            # insert the path at the beginning of the list
            self.paths.insert(0, trial)
            if len(self.paths) > self.max_paths:
                # remove the last path from the list
                self.paths.pop()
        else:
            # insert the last path at the beginning of the list
            self.paths.insert(0, self.last_path.copy_path())
            if len(self.paths) > self.max_paths:
                # remove the last path from the list
                self.paths.pop()
        trial.cycle_acc = self.cycle_acc
        trial.cycle_md = self.cycle_md
        # update PE file
        self.write_to_pe_file(trial, update_connectivity=update_connectivity)
        # and write to the order.txt file
        # self.write_to_order_file(trial, self.cycle, ptype, plen, status, gen)

    def update_infpe(self, status, path, gen, simcycle, weight,
                     start, stop, flip=False, W=None):
        """Updates the pathensemble file for the infswap move. 
        
        Parameters
        ----------
        status : str
            Status of the move. This is either "ACC" or any of the "REJ" flags.
        trial : :py:class:`Path` object
            Accepted path
        gen : int
            Generation of the move: swap (s-, s+), shoot (sh), null (00)
        simcycle : int
            Cycle number of the simulation
        weight : float
            Weight of the path
        start : int
            Index of the path in the original connection list
        end : int
            Up to where do we save the connectivity string

        """
        trial = path.connections[start][1].copy_path()
        trial.connections = path.connections  # I added this in sleepy stupor
        trial.simcycle = simcycle
        trial.gen = gen
        trial.weight = weight
        # So cycle_acc has here taken the role of 'which cycle was this 
        # extension created in?'
        trial.cycle_acc = path.connections[start][1].simcycle
        trial.cycle_md = path.connections[start][1].simcycle
        if flip:
            trial.ptype = trial.ptype[::-1]

        logger.debug(f"writing ptype {trial.ptype} to {self.id} (e,l) infpe")

        self.write_to_infpe_file(trial, start=start, stop=stop, 
                                 update_connectivity=True, W=W, flip=flip)

    def write_to_infpe_file(self, path, start, stop,
                            update_connectivity=False, W=None,
                            flip=False):
        """Format: simcycle, cycle_acc, cycle_md, ptype, plen, status,
        gen, omin, omax
        chars: 10, 10, 6, 7, 3, 2, 10decimals, 10decimals, 10
        align: right, right, right, right, right, right, right, right, right
        type: int, int, int, str, int, str, str, float, float

        """
        with open(self.infpe, "a") as f:
            f.write(PATH_FMT.format(path.simcycle, path.cycle_acc,
                                    path.cycle_md, path.ptype[0],
                                    path.ptype[1], path.ptype[2],
                                    path.plen, path.status, path.gen,
                                    path.omin, path.omax,
                                    0, 0, 0., 0, 0, path.weight))
            if W is not None:
                f.write(" " + str(W))
            
            if not update_connectivity:
                f.write("\n")
            else:
                connecstr = path.get_state_connectivity(flip=flip)
                f.write(" " + connecstr + "\n")
                # actual_connec = path.extract_from_connectivity_string(
                #     start=start, stop=stop
                # )
                # f.write(" " + actual_connec + " :FROM " +\
                #         path.connec_string + "\n")
    
    def write_to_engine(self, cycle):
        """ We write the amount of MD steps down up to now to the engine.txt
        file. This is done every cycle. Formatting: 10 positions for the cycle
        number, unlimited for the amount of MD steps.
        
        """
        with open(str(self.id) + "/engine.txt", "a+") as f:
            f.write("{:>10d} {:>100d}\n".format(cycle, self.engine.mdsteps))

    def jump_back(self, n=1):
        """Jump back n cycles in the ensemble.
        
        This removes the n last paths from the ensemble, and removes n lines 
        from the pathensemble.txt file. It is as if the last n cycles never
        happened. Why? Because force meets force.
        
        Parameters
        ----------
        n : int
            Number of cycles to jump back. Default is 1.

        """
        # remove the last n paths from the ensemble
        for i in range(n):
            self.paths.pop(0)
        # set the last_path
        self.last_path = self.paths[0]
        # remove the last n lines from the pathensemble.txt file
        remove_lines_from_file(str(self.id) + "/pathensemble.txt", n=n)
        # Update the cycle numbers
        self.cycle -= n
        # We do not update the acc_cycle, md_cycle etc

    def write_to_pe_file(self, path, update_connectivity=False):
        """Format: simcycle, cycle_acc, cycle_md, ptype, plen, status,
        gen, omin, omax
        chars: 10, 10, 6, 7, 3, 2, 10decimals, 10decimals, 10
        align: right, right, right, right, right, right, right, right, right
        type: int, int, int, str, int, str, str, float, float

        """
        with open(str(self.id) + "/pathensemble.txt", "a") as f:
            f.write(PATH_FMT.format(path.simcycle, path.cycle_acc,
                                    path.cycle_md, path.ptype[0],
                                    path.ptype[1], path.ptype[2],
                                    path.plen, path.status, path.gen,
                                    path.omin, path.omax,
                                    0, 0, 0., 0, 0, 1.))
            if not update_connectivity:
                f.write("\n")
            else:
                f.write(" " + path.connec_string + "\n")

    def set_conditions(self):
        """ Determines the start, end and cross conditions of the ensemble.

        Sets the attributes start_conditions, end_conditions and
        cross_conditions of the ensemble. These are sets containing one or more
        of the following strings: "L", "M" and "R".

        We distinguish between the following types of ensembles:
        - body_TIS: This is a regular [i^+] ensemble in (PP)TIS simulations.
        - state_A: This is the regular [0^-] ensemble in (PP)TIS simulations.
        - state_A_lambda_min_one: This is the [0^-'] ensemble in (PP)TIS
            simulations, where a lambda_{-1} is present.
        - body_PPTIS: This is a regular [i^{+-}] ensemble in PPTIS simulations.
        - PPTIS_0plusmin_primed: This is the [0^{+-}'] ensemble in PPTIS
            simulations.
        - state_B: This is the [N^-] ensemble in a snakeTIS simulation.
        - state_B_lambda_plus_one: This is the [N^-'] ensemble in a snakeTIS
            simulation, where a lambda_{N+1} is present.

        """
        # Normal ensembles don't have illegal types, and are not primed
        self.illegal_pathtypes = set()
        self.isprimed = False

        if self.ens_type == "body_TIS":
            self.start_conditions = {"L"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {"M"}

        elif self.ens_type == "state_A":
            self.start_conditions = {"R"}
            self.end_conditions = {"R"}
            self.cross_conditions = {}

        elif self.ens_type == "state_A_lambda_min_one":
            self.start_conditions = {"L", "R"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}

        elif self.ens_type == "body_PPTIS":
            self.start_conditions = {"L", "R"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {"M"}

        elif self.ens_type == "PPTIS_0plusmin_primed":
            if self.prime_both_starts:
                self.start_conditions = {"L", "R"}
            else:
                self.start_conditions = {"L"}  # no R because repptis_swap!!
            self.illegal_pathtypes = {"RMR"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}
            self.isprimed = True

        elif self.ens_type == "RETIS_0plus":
            self.start_conditions = {"L"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}

        elif self.ens_type == "state_B":
            self.start_conditions = {"L"}
            self.end_conditions = {"L"}
            self.cross_conditions = {}

        elif self.ens_type == "PPTIS_Nplusmin_primed":
            if self.prime_both_starts:
                self.start_conditions = {"L", "R"}
            else:
                self.start_conditions = {"R"}
            self.illegal_pathtypes = {"LML"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}
            self.isprimed = True

        elif self.ens_type == "state_B_lambda_plus_one":
            self.start_conditions = {"L", "R"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}

        elif self.ens_type == "iL":
            self.start_conditions = {"L"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {"M"}

        elif self.ens_type == "iR":
            self.start_conditions = {"R"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {"M"}

        elif self.ens_type == "0L":
            self.start_conditions = {"L"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}
            self.isprimed = True

        elif self.ens_type == "0R":
            self.start_conditions = {"R"}
            self.end_conditions = {"L"}
            self.cross_conditions = {}
            self.isprimed = True

        elif self.ens_type == "NL":
            self.start_conditions = {"L"}
            self.end_conditions = {"R"}
            self.cross_conditions = {}
            self.isprimed = True

        elif self.ens_type == "NR":
            self.start_conditions = {"R"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}
            self.isprimed = True            

        elif self.ens_type == "body_PPTIS_left_future":
            self.start_conditions = {"L", "R"}
            self.end_conditions = {"L"}
            self.cross_conditions = {"M"}
            self.isfuture_like = True
            self.ispast_like = False

        elif self.ens_type == "body_PPTIS_right_future":
            self.start_conditions = {"L", "R"}
            self.end_conditions = {"R"}
            self.cross_conditions = {"M"}
            self.isfuture_like = True
            self.ispast_like = False

        else:
            raise ValueError("Unknown ensemble type: {}".format(self.ens_type))

    def check_ph_in_ensemble(self, ph):
        """ Checks whether a phasepoint could be valid for of a path in the
        pathensemble.

        This is used nowhere, and rightfully so, we should only check with the
        order parameter!

        Parameters
        ----------
        ph : tuple (x, v) of floats
            Phasepoint to check

        Returns
        -------
        bool
            True if the phasepoint could be valid for a path in the
            pathensemble, False otherwise

        """
        x = ph[0]
        if self.ens_type in ["state_A"]:
            return x <= self.intfs["L"]

        elif self.ens_type in ["state_B"]:
            return x >= self.intfs["R"]

        else:
            # self.ens_type in \
            # ["body_TIS", "body_PPTIS", "state_A_lambda_min_one",
            #  "state_B_lambda_plus_one", "PPTIS_0plusmin_primed",
            #  "PPTIS_Nplusmin_primed", "body_PPTIS_left", "body_PPTIS_right",
            #  "primed_PPTIS_left", "primed_PPTIS_right", "RETIS_0plus",
            #  "body_PPTIS_left_future", "body_PPTIS_right_future"]:
            return x >= self.intfs["L"] and x <= self.intfs["R"]
        # else:
        #     raise ValueError("Unknown ensemble type: {}".format(self.ens_type))

    def check_path(self, path):
        """ Checks whether a path is valid for the ensemble.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        bool
            True if the path is valid for the ensemble, False otherwise

        """
        return self.check_start(path) and self.check_end(path) and \
            self.check_cross(path)

    def check_cross(self, path):
        """ Checks whether a path meets the cross conditions of the ensemble.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        bool
            True if the path meets the cross conditions of the ensemble,
            False otherwise

        """
        crossed = True
        # if there are no cross conditions, automatically return True
        if len(self.cross_conditions) == 0:
            return True
        # if there are cross conditions, check whether the path meets them
        omax = max([op[0] for op in path.orders])
        omin = min([op[0] for op in path.orders])
        for cross_condition in self.cross_conditions:
            crossed = crossed and \
                (omax >= self.intfs[cross_condition] and \
                 omin <= self.intfs[cross_condition])
        return crossed

    def check_start_and_end(self, path):
        """ Checks whether a path meets the start and end conditions of the
        ensemble.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        bool
            True if the path meets the start and end conditions of the ensemble,
            False otherwise

        """
        return self.check_start(path) and self.check_end(path)

    def check_start(self, path):
        """ Checks whether a path meets the start conditions of the ensemble.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        bool
            True if the path meets the start conditions of the ensemble,
            False otherwise

        """
        start = False
        op = path.orders[0][0]
        if len(self.start_conditions) == 0:
            return True
        for start_condition in self.start_conditions:
            if start_condition == "R":
                start = start or op >= self.intfs[start_condition]
            elif start_condition == "L":
                start = start or op <= self.intfs[start_condition]
            else:
                raise ValueError("Unknown start condition: {}".format(
                    start_condition))
        return start

    def check_end(self, path):
        """ Checks whether a path meets the end conditions of the ensemble.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        bool
            True if the path meets the end conditions of the ensemble,
            False otherwise

        """
        end = False
        op = path.orders[-1][0]
        if len(self.end_conditions) == 0:
            return True
        for end_condition in self.end_conditions:
            if end_condition == "R":
                end = end or op >= self.intfs[end_condition]
            elif end_condition == "L":
                end = end or op <= self.intfs[end_condition]
            else:
                raise ValueError("Unknown end condition: {}".format(
                    end_condition))
        return end

    def check_start_end_positions(self, path):
        """ Returns the start and end positions of a path.
        These are either "L", "R", or "M" for left, right, or middle,
        respectively. R denotes right of the right interface, L denotes left of
        the left interface, and M denotes between the left and right interface.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        tuple of strings
            Start and end positions of the path

        """
        startpos = "L" if path.orders[0][0] <= self.intfs["L"] else \
            "R" if path.orders[0][0] >= self.intfs["R"] else "*"
        endpos = "L" if path.orders[-1][0] <= self.intfs["L"] else \
            "R" if path.orders[-1][0] >= self.intfs["R"] else "*"
        return startpos, endpos

    def get_ptype(self, path):
        """ Returns the type of the path. This is a three letter combination of
        L, M, R and * (for left, middle, right and unknown, respectively)

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        str
            Type of the path

        """
        startpos, endpos = self.check_start_end_positions(path)
        middlepos = "M" if self.check_cross(path) else "*"
        return startpos + middlepos + endpos

    def create_initial_path(self, N = 1000):
        """Create an initial path for the ensemble.
        We will create a path that starts at one of the start_condition intfs,
        and ends in one of the end_condition intfs. We check whether there is a
        crossing condition, and if so, make sure it is met.

        Parameters
        ----------
        N : int
            Half of the number of phasepoints in the initial path

        """
        logger.info("Creating initial path for ensemble {}".format(self.name))
        # Let's make 4 general functions for the stuff above, to create 
        # LML, LMR, RML, or RMR paths. 
        L, R = self.intfs["L"], self.intfs["R"]
        M = self.intfs["M"] if self.intfs["M"] is not None else (L+R)/2

        assert L <= M and M <= R, "L <= M <= R not satisfied"
        
        special = False
        make_LMR, make_RML, make_LML, make_RMR = False, False, False, False
        make_RMR_state_A, make_RML_state_A, = False, False
        make_LML_state_B = False

        if self.ens_type in ["iR"]:
            make_RMR = True 
        elif self.ens_type in ["iL"]:
            make_LML = True
        elif self.ens_type in ["0L", "NL", "body_PPTIS",
                               "PPTIS_0plusmin_primed", "RETIS_0plus",
                               "body_TIS"]:
            make_LMR = True
        elif self.ens_type in ["0R", "NR", "PPTIS_Nplusmin_primed"]:
            make_RML = True
        elif self.ens_type in ["state_A"]:
            make_RMR_state_A = True
            special = True
        elif self.ens_type in ["state_B"]:
            make_LML_state_B = True
            special = True
        elif self.ens_type in ["state_A_lambda_min_one"]:
            make_RML_state_A = True
            special

        if make_LML:
            # we make a path from L-delta to M+delta to L-delta
            delta = (M - L) / 100
            phasepoints = [(i,0.) for i in np.linspace(L-delta, M+delta, N)]
            phasepoints += [(i,0.) for i in np.linspace(M+delta, L-delta, N)][1:]
                                                                        
        elif make_LMR:
            # we make a path from L-delta to M+delta to R+delta
            delta = (M - L) / 100
            phasepoints = [(i,0.) for i in np.linspace(L-delta, M+delta, N)]
            phasepoints += [(i,0.) for i in np.linspace(M+delta, R+delta, N)][1:]
        elif make_RML:
            # we make a path from R+delta to M-delta to L-delta
            delta = (R - M) / 100
            phasepoints = [(i,0.) for i in np.linspace(R+delta, M-delta, N)]
            phasepoints += [(i,0.) for i in np.linspace(M-delta, L-delta, N)][1:]
        elif make_RMR:
            # we make a path from R+delta to M-delta to R+delta
            delta = (R - M) / 100
            phasepoints = [(i,0.) for i in np.linspace(R+delta, M-delta, N)]
            phasepoints += [(i,0.) for i in np.linspace(M-delta, R+delta, N)][1:]
        elif make_RMR_state_A:
            # For state A ensembles, we start at the right interface
            start = self.intfs["R"]*(1 + np.sign(self.intfs["R"])*((N*10)**(-1)))
            mid = self.intfs["R"]*(1 - np.sign(self.intfs["R"])*0.1)
            stop = self.intfs["R"]*(1 + np.sign(self.intfs["R"])*((N*10)**(-1)))
            special = True
        elif make_RML_state_A:
            # For state A ensembles, we start at the right interface
            start = self.intfs["R"]*(1 + np.sign(self.intfs["R"])*((N*10)**(-1)))
            mid = (self.intfs["R"] + self.intfs["L"]) / 2
            stop = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*((N*10)**(-1)))
            special = True
        elif make_LML_state_B:
            # For state B ensembles, we start at the left interface
            start = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*((N*10)**(-1)))
            mid = self.intfs["L"]*(1 + np.sign(self.intfs["L"])*0.1)
            stop = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*((N*10)**(-1)))
            special = True
        if special:
            phasepoints1 = [(i,0.) for i in np.linspace(start, mid, N)]
            phasepoints2 = [(i,0.) for i in np.linspace(mid, stop, N)]
            phasepoints = phasepoints1 + phasepoints2[1:]
        
        if self.dim == 2:
            if self.simtype == "retis" and\
                self.ens_type not in ['state_A', 'state_A_lambda_min_one']:
                #phasepoints = np.load("retis_initpath_phs.npy")
                phasepoints = np.load("retis_shuffle_initpath.npy")
            else:
            # we add y-position of 0.5 and y-velocity of 0.0 to the phasepoints
                phasepoints = [(np.array([.1, x]), 
                                np.array([0., v])) for x, v in phasepoints]
            
        else:
            assert self.dim == 1, "dim must be 1 or 2"

        orders = [self.orderparameter.calculate(ph) for ph in phasepoints]

        path = Path(phasepoints, orders, self.id)
        assert self.check_path(path), "Initial path doesn't meet conditions"
        logger.info(f"Initial path created for ensemble {self.name}")
        for i in range(self.max_paths): 
            self.paths.append(path)
        self.last_path = path
        self.update_data("ACC", path, "ld", 0)

    def plot_min_max_distributions(self, flag="ACC"):
        """Reads the pathensemble.txt file and plots the distribution of the
        minimum and maximum order parameters of the paths in the ensemble.
        
        Parameters
        ----------
        flag : str, optional
            Flag to indicate whether to plot the distributions of the accepted
            or rejected paths. Default is "ACC".
        
        """
        if flag == "ACC":
            # we load the ACC, min and max columns (string, float, float)
            data = np.loadtxt("00"+str(self.id) + "/pathensemble.txt",
                              usecols=(7, 9, 10), dtype=str)
            data = data[data[:, 0] == "ACC"]
            data = data[:, 1:].astype(float)
        elif flag == "REJ":
            data = np.loadtxt("00"+str(self.id) + "/pathensemble.txt",
                              usecols=(7, 9, 10), dtype=str)
            data = data[data[:, 0] != "ACC"]
            data = data[:, 1:].astype(float)
        else:
            raise ValueError("Unknown flag: {}".format(flag))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.hist(data[:, 0], bins=100, density=True, label="min")
        ax.hist(data[:, 1], bins=100, density=True, label="max")
        ax.legend()
        ax.set_title("{} ensemble {}".format(flag, self.name))
        fig.show()


    def write_restart_pickle(self):
        """ Writes a pickle file containing the ensemble data.
        
        """
        with open(str(self.id) + "/restart.pkl", "wb") as f:
            pkl.dump(self, f)

    @classmethod
    def load_restart_pickle(cls, id):
        """ Loads a pickle file containing the ensemble data.
        
        """
        with open(str(id) + "/restart.pkl", "rb") as f:
            return pkl.load(f)
import os
import logging
import numpy as np
from ensemble import Ensemble
from moves import shooting_move, swap, swap_zero, repptis_swap
from snakemove import snake_move, forced_extension, snake_propagator, snake_waggle
import pickle as pkl
from funcs import (plot_paths, get_state, probe_snake_trajectory,
                   get_introspective_swap_matrix, select_submatrix,
                   permanent_prob, fastpermanent_repeat_prob)
import matplotlib.pyplot as plt

from funcs import plot_paths, overlay_paths, sample_paths, binom

import matplotlib.pyplot as plt

from moves import cut_LR_to_M, cut_extremal_phasepoints, propagate
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class extSimulation:
    """ (RE)PPTIS simulation

    Attributes
    ----------
    settings : dict
        Dictionary of simulation settings
    intfs : list of floats
        List of interfaces
    cycle : int
        Cycle number
    max_cycles : int
        Maximum number of cycles
    ensembles : list of :py:class:`Ensemble` objects
        List of ensembles
    simtype : str
        Type of simulation to perform. This is either "repptis" or "retis"
    method : str
        How is the simulation initialized? Either via loading of acceptable
        paths for each ensemble, or via a restart of a previous simulation.
    permeability: bool
        Whether we use a lambda_{-1} interface for the [0-] ensemble.
    zero_left: float
        The left interface of the [0-] ensemble. Only used for permeability.

    """
    def __init__(self, settings):
        """Initialize the Simulation object.

        Parameters
        ---------
        settings : dict
            Dictionary of settings

        """
        self.settings = settings
        self.intfs = settings["interfaces"]
        self.cycle = settings.get("cycle", 0)
        self.max_cycles = settings.get("max_cycles", 1000000)
        self.simtype = settings["simtype"]
        self.method = settings["method"]
        self.permeability = settings.get("permeability", False)
        self.zero_left = settings.get("zero_left", None)
        self.simtype = settings.get("simtype", "retis")
        self.p_shoot = settings.get("p_shoot", 0.9)
        # Wheter or not we include a state B ensemble (like [0-] on the right)
        self.include_stateB = settings.get("include_stateB", False)
        # Whether or not we let [0+'] (and [N+-']) paths start from both L and R
        self.prime_both_starts = settings.get("prime_both_starts", False)
        # The amount of levels in each pathensemble
        self.Nl = settings.get("Nl", 5)
        logger.info("Initializing the {} simulation.".format(self.simtype))
        # Making the ensembles, Nl levels
        self.ensembles = [[] for _ in range(self.Nl)]
        self.Next = settings.get("Next", 0)
        self.yoot = settings.get("yoot", False)
        self.endpoints_only = settings.get("endpoints_only", True)
        self.flipboth = settings.get("flipboth", False)
        self.prob_onW_insteadofS = settings.get("prob_onW_insteadofS", False)
        self.prob_onWinv_insteadofS = settings.get("prob_onWinv_insteadofS", False)
        self.binomial = settings.get("binomial", False)
        self.random_shoots = settings.get("random_shoots", False)
        self.invert_W = settings.get("invert_W", False)
        self.multilevelke = settings.get("multilevelke", False)

        if self.method != "restart":
            # Create the ensembles
            logger.info("Making the ensembles")
            for i in range(self.Nl):
                self.create_ensembles(level=i)
            logger.info("Done making the ensembles")

            logger.info("Creating dummy initial paths for the ensembles")
            for level in self.ensembles:
                for ens in level:
                    ens.create_initial_path()
            logger.info("Done creating dummy initial paths for the ensembles")
        else:
            # load the ensembles from restart pickles
            logger.info("Loading restart pickles for the ensembles")
            self.load_ensembles_from_restart()
            logger.info("Done loading restart pickles for ensembles")

        # How many ensembles do we have
        self.Ne = len(self.ensembles[0])
        self.set_swap_partitions()

    def do_extension_move(self, l, e, ext):
        """Perform an extension move in a chosen ensemble.

        """
        path_to_extend = self.ensembles[l][e].last_path
        ens = self.ensembles[l][e]
        logger.debug(f"Shooting in ensemble {l}-{e} (l-e)")
        ens_from_id = ens.last_path.connections[-1][0]
        ens_from = self.ensembles[ens_from_id[1]][ens_from_id[0]]
        trial = ens.last_path.connections[-1][1]
        assert ens_from.get_ptype(trial) == ens.last_path.connections[-1][2]
        msg = f"Extending path in {ens.id}, which had reached {ens_from.id}"
        msg += f"for an extra {ext} steps"
        logger.info(msg)
        trials, statuses = [trial], ["ACC"]
        connections = []
        for step in range(ext):
            logger.debug("--> Step {}".format(step))
            reverse, ens_to = snake_waggle(self.ensembles, ens_from,
                                           propdir="forwards",
                                           path=trials[-1], multilevel=True)
            tstatus, ttrial = snake_propagator(ens_from, ens_to, reverse,
                                               forcing=False, path=trials[-1])
            trials.append(ttrial)
            statuses.append(tstatus)
            connections.append((ens_to.id, ttrial, ens_to.get_ptype(ttrial)))
            ens_from = ens_to
            logger.debug("Wiggle status: {}".format(tstatus))

        # if all statuses are 'ACC', we update the original ensemble
        if all([s == "ACC" for s in statuses]):
            logger.info("All wiggles were successful, updating data")
            path_to_extend.extend_connectivity_string(connections)
            # and we update the data of each extended path to the same 
            # connectivity string. But NOT the connections.
            for path in [connec[1] for connec in path_to_extend.connections]:
                path.connec_string = path_to_extend.connec_string
        else:
            logger.info("Something went wrong, here is the data:")
            for t, s in zip(trials, statuses):
                logger.info("Trial: {}, Status: {}".format(t, s))
            # stop the simulation
            raise ValueError("Something went wrong with the extension move")      

    def do_shooting_move(self, l=None, e=None, ext=0):
        """Perform a shooting move in a chosen ensemble.

        """
        #self.cycle += 1
        e = np.random.choice(self.Ne) if e is None else e
        l = np.random.choice(self.Nl) if l is None else l
        ens = self.ensembles[l][e]
        status, trial = shooting_move(ens)
        logger.info("Shooting move in {}-{} resulted in {}".format(
                     ens.name, ens.level, status))
        if status != "ACC":
            ens.update_data(status, trial, "sh", self.cycle)
            return
        logger.info("Forcibly extending the path for {} steps".format(ext))

        ens_from = ens
        trial_copy = trial.copy_path()
        ens_from.update_data(status, trial_copy, "ex", self.cycle)
        trials, statuses = [trial_copy], [status]  # copy s.t. 0 connecs
        connections = [(ens.id, trial_copy, ens.get_ptype(trial))]
        for step in range(ext):
            logger.debug("--> Step {}".format(step))
            reverse, ens_to = snake_waggle(self.ensembles, ens_from,
                                           propdir="forwards",
                                           path=trials[-1], multilevel=True)
            tstatus, ttrial = snake_propagator(ens_from, ens_to, reverse,
                                               forcing=False, path=trials[-1])
            # let's update the data of the path
            ens_to.update_data(tstatus, ttrial, "ex", self.cycle)
            trials.append(ttrial)
            statuses.append(tstatus)
            connections.append((ens_to.id, ttrial, ens_to.get_ptype(ttrial)))
            ens_from = ens_to
            logger.debug("Wiggle status: {}".format(tstatus))
        
        # if all statuses are 'ACC', we update the original ensemble
        if all([s == "ACC" for s in statuses]):
            logger.info("All wiggles were successful, updating data")
            trial.connections = connections
            # TODO: for now we just set the connectivity string of the 
            # connected paths to the same connectivity string.
            trial.set_connectivity_string(connections)
            for tttrial in trials:
                tttrial.set_connectivity_string(connections)
            ens.update_data(status, trial, "sh", self.cycle,
                            update_connectivity=True)
        else:
            logger.info("Something went wrong, here is the data:")
            for t, s in zip(trials, statuses):
                logger.info("Trial: {}, Status: {}".format(t, s))
            # stop the simulation
            raise ValueError("Something went wrong with the shooting move")
        
    def do_random_shooting_move(self, l=None, e=None, ext=0):
        """Perform a shooting move in a chosen ensemble.

        """
        #self.cycle += 1
        ens = self.ensembles[l][e]
        status, trial = shooting_move(ens)
        logger.info("Shooting move in {}-{} resulted in {}".format(
                     ens.name, ens.level, status))
        if status != "ACC":
            ens.update_data(status, trial, "sh", self.cycle)
            return
        
        #ext_fw = np.random.randint(0, ext+1)
        # This is wrong, we don't want uniform random, but a binomial 
        # distribution with p=0.5
        ext_fw = np.random.binomial(ext, 0.5)
        logger.info(f"Forcibly extending the path FW for {ext_fw} steps")
        ens_from = ens
        trial_copy = trial.copy_path()
        ens_from.update_data(status, trial_copy, "ex", self.cycle)
        trials, statuses = [trial_copy], [status]  # copy s.t. 0 connecs
        connections = [(ens.id, trial_copy, ens.get_ptype(trial))]
        for step in range(ext_fw):
            logger.debug("--> Step {}".format(step))
            reverse, ens_to = snake_waggle(self.ensembles, ens_from,
                                           level = 0, propdir="forwards",
                                           path=trials[-1], multilevel=True)
            tstatus, ttrial = snake_propagator(ens_from, ens_to, reverse,
                                               level=0,
                                               forcing=False, path=trials[-1])
            # let's update the data of the path
            ens_to.update_data(tstatus, ttrial, "ex", self.cycle)
            trials.append(ttrial)
            statuses.append(tstatus)
            connections.append((ens_to.id, ttrial, ens_to.get_ptype(ttrial)))
            ens_from = ens_to
            logger.debug("Wiggle status: {}".format(tstatus))
        
        logger.info(f"Forcibly extending the path BW for {ext-ext_fw} steps")
        trials_bw, statuses_bw = [trial_copy], [status]
        ens_from = ens
        connections_bw = []
        for step in range(ext - ext_fw):
            logger.debug("--> Step {}".format(step))
            reverse, ens_to = snake_waggle(self.ensembles, ens_from, level=0,
                                           propdir="backwards",
                                           path=trials_bw[-1], multilevel=True)
            logger.debug(f"BW wiggle should have reverse -1: {reverse}")
            tstatus, ttrial = snake_propagator(ens_from, ens_to, reverse,
                                               level=0, forcing=False,
                                               path=trials_bw[-1])
            # let's update the data of the path
            ens_to.update_data(tstatus, ttrial, "ex", self.cycle)
            trials_bw.append(ttrial)
            statuses_bw.append(tstatus)
            connections_bw.append((ens_to.id, ttrial, ens_to.get_ptype(ttrial)))
            ens_from = ens_to
            logger.debug("Wiggle status: {}".format(tstatus)) 

        # if all statuses are 'ACC', we update the original ensemble
        if all([s == "ACC" for s in statuses]):
            logger.info("All wiggles were successful, updating data")
            connections_bw = connections_bw[::-1]
            trial.connections = connections_bw + connections
            # TODO: for now we just set the connectivity string of the 
            # connected paths to the same connectivity string.
            trial.set_connectivity_string(connections_bw + connections)
            for tttrial in trials + trials_bw:
                tttrial.set_connectivity_string(connections_bw + connections)
            ens.update_data(status, trial, "sh", self.cycle,
                            update_connectivity=True)
        else:
            logger.info("Something went wrong, here is the data:")
            for t, s in zip(trials+trials_bw, statuses):
                logger.info("Trial: {}, Status: {}".format(t, s))
            # stop the simulation
            raise ValueError("Something went wrong with the shooting move")

    def do_infinity_move(self, shoots=True):
        """We perform an infinity move. This entails:
            1) Perform shooting moves in all the ensembles, at all levels
            2) Choose a swapping frame that partitions the ensembles into Np
               non-empty sets that cover all the ensembles.
            3) Perform infinite swaps in each swapping frame
            4) Extend the paths that are connected and lack memory.
            5) Update the pathensemble data using the results of the frames
            6) Select paths from the swapping frames and appoint them randomnly
               to the ensembles, such that each ensemble has exactly one path.
        """
        # 1) Perform shooting moves in all the ensembles, at all levels
        if shoots:
            for i in range(self.Nl):
                for j in range(self.Ne):
                    if self.random_shoots:
                        logger.info(f"Doing random shooting move in {i}-{j}")
                        self.do_random_shooting_move(l=i, e=j, ext=self.Next)
                    else:
                        logger.info(f"Doing shooting move in {i}-{j}")
                        self.do_shooting_move(l=i, e=j, ext=self.Next)
        # 2) Choose a partition from the self.partitions list
        partition = self.partitions[np.random.randint(0, len(self.partitions))]
        # 3) Perform infinite swaps in each swapping frame of the partition
        if self.Next == 0:
            S, W = np.eye(self.Ne*self.Nl),\
                    np.eye(self.Ne*self.Nl, dtype=np.float64)
        else:
            S, W = get_introspective_swap_matrix(
                self.ensembles,
                endpoints_only=self.endpoints_only,
                binomial=self.binomial,
                invert=self.invert_W
                )

        for iframe, frame in enumerate(partition):
            logger.info(f"DOING FRAME {iframe}")
            connected_swaps = []
            # select the frame out of the S matrix
            Sframe = select_submatrix(S, cols=frame[0], rows=frame[1])
            Wframe = select_submatrix(W, cols=frame[0], rows=frame[1])
            # perform the infinite swap
            #P = fastpermanent_repeat_prob(Sframe, r=1) # TODO: make r a setting
            P = np.abs(permanent_prob(Wframe))
            # if P[i,j] is approx 0, we set it to 0
            # P[P < 1e-10] = 0.

            #P = np.abs(permanent_prob(Sframe.astype(np.float64)))
            # P = np.eye(len(Sframe)//2)/2
            # Wframe = np.eye(len(Sframe)//2, dtype=int)
            # Sframe = np.eye(len(Sframe)//2)
            # P = np.vstack([np.hstack([P, P])]*2)
            # Wframe = np.vstack([np.hstack([Wframe, Wframe])]*2)
            # Sframe = np.vstack([np.hstack([Sframe, Sframe])]*2)
            # print element in 0.xz format
            msg = "\n".join([" ".join(["{:.2f}".format(x) if x>0.0001 else "----" for x in row])\
                             for row in Sframe])
            logger.info("S matrix:\n{}".format(msg))
            msg2 = "\n".join([" ".join(["{:.2f}".format(x) if x>0.0001 else "----" for x in row])\
                             for row in P])
            msg3 = "\n".join([" ".join(["{:.2f}".format(x) if x>0.0001 else "----" for x in row])\
                                for row in Wframe])
            logger.info("W matrix:\n{}".format(msg3))

            logger.info("P matrix:\n{}".format(msg2))
            # 4a) Scan the connectivity of swaps, and see if we need to extend
            # for i, iF in enumerate(frame[0]):
            #     lp, ep = divmod(iF, self.Ne)
            #     logger.debug(f"Looking at path of ensemble {lp}-{ep}")
            #     for j, jF in enumerate(frame[1]):
            #         le, ee = divmod(jF, self.Ne)
            #         logger.debug(f"checking ensemble {le}-{ee}")
            #         ens_from = self.ensembles[lp][ep]  # path we're swapping
            #         path_from = ens_from.last_path
            #         ens_to = self.ensembles[le][ee]  # ens we swap to
            #         weight = P[i, j]
            #         if weight > 0:
            #             # assert that there is an actual connection
            #             csum = np.sum([connec[0][0] == ee\
            #                            for connec in path_from.connections])
            #             assert csum > 0, "No connection found"
            #             # Now, detect which path is connected to the path in the
            #             # ensemble, and update the data.
            #             for ic, connec in enumerate(path_from.connections):
            #                 constringlen =\
            #                     path_from.get_connectivity_string_length()
            #                 if connec[0][0] == ee:
            #                     msg = f"Connection with connecstring"
            #                     msg += f" {connec[1].connec_string}"
            #                     logger.info(msg)
            #                     if constringlen < ic + self.Next + 1:
            #                         # Extend this path by the difference
            #                         ext = ic + self.Next - constringlen + 1
            #                         connected_swaps.append((lp, ep, ext))
            #                         msg = f"Have to extend this one by {ext}"
            #                         logger.debug(msg)
            # 4b) Perform the required memory extensions.
            # But we don't want to extend the same path multiple times, so
            # first we look for the longest extension required for each path.
            # extension_tuples = defaultdict(lambda: float("-inf"))
            # for l, e, ext in connected_swaps:
            #     extension_tuples[(l, e)] = max(extension_tuples[(l, e)],
            #                                      ext)
            # extension_tuples = [(l, e, ext) for (l, e), ext in\
            #                     extension_tuples.items()]
            # 4c) Perform the required memory extensions
            # for l, e, ext in extension_tuples:
            #     if ext > 0:
            #         self.do_extension_move(l, e, ext)
            #     else: print("No extensino needed.")
            # 5) Update the pathensemble data using the results of the frames
            for i, iF in enumerate(frame[0]):
                lp, ep = divmod(iF, self.Ne)
                logger.debug(f"Upd ensembs where path {lp}-{ep} is connected")
                temp = self.ensembles[lp][ep].last_path
                logger.debug(f"connecs in {lp}-{ep}:{temp.connections}")
                for j, jF in enumerate(frame[1]):
                    le, ee = divmod(jF, self.Ne)
                    logger.debug("Checking ensemble {}-{}".format(le, ee))
                    weight = P[i, j]
                    if weight > 0.00001:
                        if self.endpoints_only:
                            inv_connecs = [temp.connections[0]] +\
                                          [temp.connections[-1]]
                            inv_ids = [0, len(temp.connections)-1]
                        else:
                            inv_connecs = temp.connections
                            inv_ids = [i for i in range(len(temp.connections))]
                        for ic, connec in zip(inv_ids, inv_connecs):
                            if connec[0][0] == ee: # and (ep == ee):
                                msg = f"Updating {lp}-{ep} p into ensemble {le}-{ee},"
                                msg += f"start: {ic}, stop: {ic+self.Next}"
                                logger.debug(msg)
                                assert Wframe[i, j] != 0, "Weight is zero"
                                logger.debug(f"With weight: {P[i,j]}")
                                start = ic
                                stop = ic + self.Next 
                                #flip = False if ic == 0 else True
                                # for flipboths, this was true!!
                                # flip = True
                                flip = False
                                self.ensembles[le][ee].update_infpe(
                                    "ACC", temp, "oo", self.cycle,
                                    #weight/Sframe[i,j],  # use this if no break
                                    weight,
                                    start=start, stop=stop, flip=flip,
                                    W = W[i,j]
                                )
                                if self.flipboth:
                                    logger.debug("Also flipping the path")
                                    self.ensembles[le][ee].update_infpe(
                                        "ACC", temp, "of", self.cycle,
                                        weight,
                                        #weight/Sframe[i,j],  # use if no break
                                        start=start, stop=stop, flip=not flip,
                                        W = W[i,j]
                                    )
                                break  # we only update once now.
            # 6) Select paths from the swapping frames and appoint them to the
            # ensembles, such that each ensemble has exactly one path.
            # We go column by column (or row by row?) in the P matrix, and 
            # select a path at random, with weights given by the P column.
            # Before going to the next column, we renormalize the weights
            # of the rows (without including the weights of the already
            # sampled ensemble-columns).
            ids, _ = sample_paths(P, Wframe, binomial=self.binomial)
            # Now, for the ensembles in the frame, we assign the paths 
            # according to the choices. We got to this in a two-step process,
            # as we may be overwriting the last path of an ensemble, which
            # which we will need to assign to another ensemble.
            # We could use a temp_path, but let's for now just use the 
            # distinction between last_path and paths[0] (which should 
            # actually always be the same, though...) Okay, let's not do it 
            # then. Let's also assign copies of the paths to the last_path
            # and temp_path, because I'm a scared little boy.
            for i, id in zip(range(len(frame[0])), ids):
                lp, ep = divmod(frame[0][i], self.Ne)
                le, ee = divmod(frame[1][id], self.Ne)
                ens_from = self.ensembles[lp][ep]
                ens_to = self.ensembles[le][ee]
                path_from = ens_from.last_path
                logger.debug("Assigning path (l,e) {}-{} to (l,e) {}-{}".format(
                    lp, ep, le, ee)
                )
                # loop through the connected paths in the ensembles last path,
                # and take the choice'th path of ensemble ee
                possibilities, possweights = [], []
                for ic, connec in enumerate(path_from.connections):
                    if connec[0][0] == ee:
                            logger.debug(f"Possibility at {ic}")
                            possibilities.append(ic)
                            possweights.append(binom(ic, len(path_from.connections)-1)[0])
                possweights = np.array(possweights)
                possweights = possweights / np.sum(possweights)
                logger.debug(f"Possibilities' probabilities: {possweights}")
                choice = np.random.choice(possibilities, p=possweights)
                ens_to.temp_path = path_from.connections[choice][1]
                # BUT WE KEEP THE ORIGNAL CONNECTIONS
                # IF NOT, YOU DEFINITELY NEED MORE DETAILED 
                # ASSIGNMENTS TO BALANCE THE PATHS, OK????
                logger.debug("Temp path: {}".format(ens_to.temp_path))
                logger.debug("connections: {}".format(ens_to.temp_path.connections))
                ens_to.temp_path.connections =\
                    ens_from.last_path.connections


            # in the second loop, we actually assign the paths
            logger.debug("Assigning the paths to the ensembles.")
            for eid in frame[1]:
                le, ee = divmod(eid, self.Ne)
                ens = self.ensembles[le][ee]
                # This is though stuff, if you don't assign the connections
                # specifically, they are lost in last_path after temp_path
                # is relieved from memory.
                ens.last_path = ens.temp_path.copy_path()
                ens.last_path.connections = ens.temp_path.connections
                ens.temp_path = None
                ens.paths[0] = ens.last_path
                if len(ens.paths) > ens.max_paths:
                    ens.paths.pop()

        logger.info("Infinity move DONE!!")

    def set_swap_partitions(self):
        """Set the swapping frames for the infinity move. For now, we assume that 
        we are going to use 4 levels, and 12 ensembles.
        Frame 1: split between upper two and lower two levels
        Frame 2: split between ensembles 0-5 and 6-11.
        
        A partition consists of a list of frames. A frame consists of two 
        lists, where the first is the indices of the ensembles, and the second
        is the indices of the levels.

        Deinen partitionen sind nicht gut. 
        My partitions should partition the (Ne x Nl) x (Ne x Nl) matrix into
        non-empty sets that cover all the ensembles. 
        """
        Ne, Nl = self.Ne, self.Nl
        if self.multilevelke and self.Nl == 4:
            # Frame 1: 
            # A: 0-->(Ne*Nl)//2 
            # B: (Ne*Nl)//2-->(Ne*Nl)
            frame1A = ([i for i in range((Ne*Nl)//2)],
                    [i for i in range((Ne*Nl)//2)])
            frame1B = ([i for i in range((Ne*Nl)//2, Nl*Ne)],
                    [i for i in range((Ne*Nl)//2, Nl*Ne)])
            partition1 = [frame1A, frame1B]

            # Frame 2: 
            # A: 0*Ne --> 0*Ne + Ne//2 union 1*Ne --> 1*Ne + Ne//2 union
            # ... union (Nl-1)*Ne --> (Nl-1)*Ne + Ne//2
            # B: 0*Ne + Ne//2 --> 0*Ne + Ne union 1*Ne + Ne//2 --> 1*Ne + Ne 
            # union ... union (Nl-1)*Ne + Ne//2 --> (Nl-1)*Ne + Ne
            frame2Acomp, frame2Bcomp = [], []
            for i in range(Nl):
                frame2Acomp += [i*Ne + j for j in range(Ne//2)]
                frame2Bcomp += [i*Ne + j for j in range(Ne//2, Ne)]
            frame2A = (frame2Acomp, frame2Acomp)
            frame2B = (frame2Bcomp, frame2Bcomp)
            partition2 = [frame2A, frame2B]    


            # partition three splits by for, and will have 4 frames
            # A: 0*Ne --> 0*Ne + Ne//4 union 1*Ne --> 1*Ne + Ne//4 union
            # ... union (Nl-1)*Ne --> (Nl-1)*Ne + Ne//4     etc
            # B: 0*Ne + Ne//4 --> 0*Ne + Ne//2 union 1*Ne + Ne//4 --> 1*Ne + Ne//2
            # ... union (Nl-1)*Ne + Ne//4 --> (Nl-1)*Ne + Ne//2
            # C: 0*Ne + Ne//2 --> 0*Ne + 3*Ne//4 union 1*Ne + Ne//2 --> 1*Ne + 3*Ne//4
            # ... union (Nl-1)*Ne + Ne//2 --> (Nl-1)*Ne + 3*Ne//4
            # D: 0*Ne + 3*Ne//4 --> 0*Ne + Ne union 1*Ne + 3*Ne//4 --> 1*Ne + Ne
            frame1Acomp, frame1Bcomp, frame1Ccomp, frame1Dcomp = [], [], [], []
            for i in range(Nl):
                frame1Acomp += [i*Ne + j for j in range(Ne//4)]
                frame1Bcomp += [i*Ne + j for j in range(Ne//4, Ne//2)]
                frame1Ccomp += [i*Ne + j for j in range(Ne//2, 3*Ne//4)]
                frame1Dcomp += [i*Ne + j for j in range(3*Ne//4, Ne)]
                
            frame1A = (frame1Acomp, frame1Acomp)
            frame1B = (frame1Bcomp, frame1Bcomp)
            frame1C = (frame1Ccomp, frame1Ccomp)
            frame1D = (frame1Dcomp, frame1Dcomp)
            partition3 = [frame1A, frame1B, frame1C, frame1D]

            # partition 4 just goes 0-->(Ne*Nl)//4, (Ne*Nl)//4-->2*(Ne*Nl)//4, etc
            frame4A = ([i for i in range((Ne*Nl)//4)],
                        [i for i in range((Ne*Nl)//4)])
            frame4B = ([i for i in range((Ne*Nl)//4, (Ne*Nl)//2)],
                        [i for i in range((Ne*Nl)//4, (Ne*Nl)//2)])
            frame4C = ([i for i in range((Ne*Nl)//2, 3*(Ne*Nl)//4)],
                        [i for i in range((Ne*Nl)//2, 3*(Ne*Nl)//4)])
            frame4D = ([i for i in range(3*(Ne*Nl)//4, Ne*Nl)],
                        [i for i in range(3*(Ne*Nl)//4, Ne*Nl)])
            partition4 = [frame4A, frame4B, frame4C, frame4D]

            # partition 5, we split in three groups (all levels, only
            # 4 ensembles)
            # so for first frame, we have 0-[0,1,2,3], 1-[0,1,2,3], etc
            # which is equal to [0, 1, 2, 3, Ne, Ne+1, Ne+2, Ne+3, ...]
            # then the second frame will have 
            # [4, 5, 6, 7, Ne+4, Ne+5, Ne+6, Ne+7, ...]
            frame5Acomp, frame5Bcomp, frame5Ccomp = [], [], []
            for i in range(Nl):
                frame5Acomp += [i*Ne + j for j in range(4)]
                frame5Bcomp += [i*Ne + j for j in range(4, 8)]
                frame5Ccomp += [i*Ne + j for j in range(8, Ne)]
            frame5A = (frame5Acomp, frame5Acomp)
            frame5B = (frame5Bcomp, frame5Bcomp)
            frame5C = (frame5Ccomp, frame5Ccomp)
            partition5 = [frame5A, frame5B, frame5C]
            


            self.partitions = [partition5, partition4]

            # Now we make 
        else:
            frame = ([i for i in range(Ne*Nl)],
                     [i for i in range(Ne*Nl)])
            self.partitions = [[frame]]


    def do_swap_moves(self):
        """ Perform swap moves in all the ensembles, where we swap the
        ensembles with their next neighbours.

        Scheme 1:
        null move in ensemble 0 and swap 1 with 2, 2 with 3, etc. If there is
        an even number of ensembles, we also do a null move in the last ensemble

        Scheme 2:
        Swap 0 with 1, 1 with 2, etc. If there is an odd number of ensembles,
        we do a null move in the last ensemble

        """
        self.cycle += 1

        scheme = np.random.choice([1, 2])
        odd = False if len(self.ensembles) % 2 == 0 else True
        if scheme == 1:
            self.do_null_move(0, "00")
            if not odd:
                self.do_null_move(-1, "00")
            for i in range(1, len(self.ensembles) - 1, 2):
                self.do_swap_move(i)
        elif scheme == 2:
            if odd:
                self.do_null_move(-1, "00")
            for i in range(0, len(self.ensembles) - 1, 2):
                self.do_swap_move(i)

    def do_swap_move(self, i):
        """ Perform a swap move in ensemble i.

        Parameters
        ----------
        i : int
            Index of the ensemble in which to perform the swap move.

        """
        if i == 0:
            status, trial1, trial2 = swap_zero(self.ensembles)
            logger.info("Swap move {} <-> {} resulted in {}".format(
                self.ensembles[i].name, self.ensembles[i+1].name, status))
            self.ensembles[i].update_data(status, trial1, "s+", self.cycle)
            self.ensembles[i+1].update_data(status, trial2, "s-", self.cycle)
            return

        if self.simtype == "retis":
            status, trial1, trial2 = swap(self.ensembles, i)
            logger.info("Swap move {} <-> {} resulted in {}".format(
                self.ensembles[i].name, self.ensembles[i+1].name, status))
            self.ensembles[i].update_data(status, trial1, "s+", self.cycle)
            self.ensembles[i+1].update_data(status, trial2, "s-", self.cycle)

        elif self.simtype == "repptis":
            status, trial1, trial2 = repptis_swap(self.ensembles, i)
            logger.info("Swap move {} <-> {} resulted in {}".format(
                self.ensembles[i].name, self.ensembles[i+1].name, status))
            self.ensembles[i].update_data(status, trial1, "s+", self.cycle)
            self.ensembles[i+1].update_data(status, trial2, "s-", self.cycle)


    def do_null_move(self, i, gen="00"):
        """ Perform a null move in ensemble i.

        Parameters
        ----------
        i : int
            Index of the ensemble in which to perform the null move.

        """
        self.ensembles[i].update_data("ACC", self.ensembles[i].last_path,
                                      gen, self.cycle)


    def create_ensembles(self, level=0):
        """ Create all the ensembles for the simulation (level=None), or for
        a particular level of the simulation (level=int). 

        Parameters
        ----------
        level : int
            Level at which to create the ensembles. 
        """
        # First we make the zero minus ensemble
        ens_set = {}
        ens_set["id"] = (0,level)
        ens_set["max_len"] = self.settings["max_len"]
        ens_set["simtype"] = self.simtype
        ens_set["temperature"] = self.settings["temperature"]
        ens_set["friction"] = self.settings["friction"]
        ens_set["dt"] = self.settings["dt"]
        ens_set["prime_both_starts"] = self.prime_both_starts
        ens_set["max_paths"] = self.settings["max_paths"]
        ens_set["level"] = level
        ens_set["mass"] = self.settings["mass"]
        ens_set["dim"] = self.settings["dim"]

        if self.permeability:
            assert self.zero_left is not None, "No zero_left for permeability"
            ens_set["intfs"] = {"L": self.zero_left,
                                "M": None,  # Not needed for permeability
                                "R": self.intfs[0]}
            ens_set["ens_type"] = "state_A_lambda_min_one"
            ens_set["name"] = "[0-']"
            logger.info("Making ensemble {}".format(ens_set["name"]))
        else:
            ens_set["intfs"] = {"L": -np.infty,
                                "M": None,  # Not even defined for 0-
                                "R": self.intfs[0]}
            ens_set["ens_type"] = "state_A"
            ens_set["name"] = "[0-]"
            logger.info("Making ensemble {}".format(ens_set["name"]))
        self.ensembles[level].append(Ensemble(ens_set))

        # Then we make the zero plus ensemble
        ens_set["id"] = (1, level)
        if self.simtype == "repptis":
            ens_set["intfs"] = {"L": self.intfs[0],
                                "M": None,  # Not even defined for 0+ or 0+-'
                                "R": self.intfs[1]}
            ens_set["ens_type"] = "PPTIS_0plusmin_primed"
            ens_set["name"] = "[0+-']"
        elif self.simtype == "retis":
            ens_set["intfs"] = {"L": self.intfs[0],
                                "M": None,  # Not even defined for 0+ or 0+-'
                                "R": self.intfs[-1]}
            ens_set["ens_type"] = "RETIS_0plus"
            ens_set["name"] = "[0+]"
        logger.info("Making ensemble {}".format(ens_set["name"]))
        self.ensembles[level].append(Ensemble(ens_set))

        # Then we make the body ensembles
        for i in range(0, len(self.intfs) - 2):
            ens_set["id"] = (i + 2, level)
            if self.simtype == "repptis":
                ens_set["intfs"] = {"L": self.intfs[i],
                                    "M": self.intfs[i + 1],
                                    "R": self.intfs[i + 2]}
                ens_set["ens_type"] = "body_PPTIS"
                ens_set["name"] = f"[{i+1}+-]"
            elif self.simtype == "retis":
                ens_set["intfs"] = {"L": self.intfs[0],
                                    "M": self.intfs[i+1],
                                    "R": self.intfs[-1]}
                ens_set["ens_type"] = "body_TIS"
                ens_set["name"] = f"[{i+1}+]"
            logger.info("Making ensemble {}".format(ens_set["name"]))
            self.ensembles[level].append(Ensemble(ens_set))

        # Dealing with state B ensembles, if requested
        if self.include_stateB:
            logger.info("Making state B ensembles")
            # Create the N+-' ensemble
            msg = "State B ensembles only implemented for repptis"
            assert self.simtype == "repptis", msg
            ens_set["id"] = (len(self.intfs), level)
            ens_set["intfs"] = {"L": self.intfs[-2],
                                "M": None,  # Not even defined for N+-'
                                "R": self.intfs[-1]}
            ens_set["ens_type"] = "PPTIS_Nplusmin_primed"
            ens_set["name"] = f"[{len(self.intfs)-1}+-']"
            logger.info("Making ensemble {}".format(ens_set["name"]))
            self.ensembles[level].append(Ensemble(ens_set))
            # Create the N- ensemble
            ens_set["id"] = (len(self.intfs) + 1, level)
            ens_set["intfs"] = {"L": self.intfs[-1],
                                "M": None,  # Not even defined for N-
                                "R": np.infty}
            ens_set["ens_type"] = "state_B"
            ens_set["name"] = f"[{len(self.intfs)-1}-]"
            logger.info("Making ensemble {}".format(ens_set["name"]))
            self.ensembles[level].append(Ensemble(ens_set))


    def load_ensembles_from_restart(self):
        """ Load the restart pickles for each ensemble. """
        for i in range(len(self.interfaces)):
            self.ensembles.append(Ensemble.load_restart_pickle(i))

    def save_simulation(self, filename):
        """ Save the simulation to a pickle file. """
        # first check if the file exists, 
        # if so, we rename it to a backup file, which we delete after
        # saving the new file successfully
        if os.path.exists(filename):
            os.rename(filename, filename + ".bak")
        with open(filename, "wb") as f:
            pkl.dump(self, f)
        # delete the backup file
        if os.path.exists(filename + ".bak"):
            os.remove(filename + ".bak")

    def write_to_enginetxt(self):
        """ For each ensemble, we write the amount of MD steps that have been 
        done up to now. So we write the cycle number and amount of MD steps"""
        for level in self.ensembles:
            for ens in level:
                ens.write_to_engine(self.cycle)

    def run(self):
        p_shoot = self.p_shoot
        while self.cycle < self.max_cycles:
            if self.cycle % 500 == 0:
                print("Cycle {}".format(self.cycle))
            logger.info("-" * 80)
            logger.info("Cycle {}".format(self.cycle))
            logger.info("-" * 80)
            if np.random.rand() < p_shoot:
                self.do_shooting_move(ext=0)
            else:
                self.do_swap_moves()
            self.cycle += 1

    def run_inf(self, restart=False):
        logger.info("Running an infinity simulation with {} levels".format(
            self.Nl))
        logger.info("--------------------------------------------------------")
        logger.info("Shooting in each ensemble until we have 1 ACC path.")
        if not restart:
            for l, level in enumerate(self.ensembles):
                for e, ens in enumerate(level):
                    while ens.last_path.gen == 'ld':
                        self.do_shooting_move(l=l, e=e, ext=self.Next)
                    logger.info("Made an ACC path in ensemble {}-{}".format(l, e))
        logger.info("-----------------------------------------------")
        logger.info("Starting the infinity moves.")
        while self.cycle < self.max_cycles:
            self.cycle += 1
            logger.info("-----------------------------------------------")
            logger.info("--- Cycle {} ---".format(self.cycle))
            logger.info("-----------------------------------------------")
            self.do_infinity_move()
            self.write_to_enginetxt()
            if self.cycle % 1000 == 0:
                logger.info(f"Saving the simulation at cycle {self.cycle}")
                self.save_simulation("inf_sim.pkl")



    @classmethod
    def load_simulation(cls, filename):
        """Load a simulation object from a pickle file.
        
        Parameters
        ----------
        filename : str
            Name of the pickle file to load.
        """
        with open(filename, "rb") as f:
            return pkl.load(f)

    
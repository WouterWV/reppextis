import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class Path:
    """ Representation of a path in PPTIS. We keep a global counter of the
    number of paths that have been created, so that each path has a unique
    identifier.

    Attributes
    ----------
    phasepoints : list of tuples (x, v) of floats
        List of phasepoints in the path
    orders : list of floats
        List of order parameters for each phasepoint
    path_id : int
        Unique identifier of the path
    ens_id : int
        Unique identifier of the ensemble to which the path belongs

    """
    path_counter = 0

    def __init__(self, phasepoints=None, orders=None, ens_id=None,
                 connections=None, connec_string=None, gen=None,
                 status=None, simcycle=None,
                 omin=None, omax=None, ptype=None, plen=None, cycle_acc=None,
                 cycle_md=None):
        """Initialize the Path object.

        Parameters
        ----------
        phasepoints : list
            List of phasepoints in the path
        ens_id : int
            Unique identifier of the ensemble to which the path belongs
        path_id : int
            Unique identifier of the path
        connections : list of tuples (int, :py:class:`Path` object, str)
            Making it easy on myself, and giving the path itself instead of 
            a linker to it. TODO: change this ;)
            Elements are tuples (i, p, t), where p is a path of type t in 
            ensemble i
        connec_string : str
            String representation of the connectivity of the path. This can 
            differ from the actual connecttions FW in the path, because of 
            the swap move detailed balance condition.

        """
        self.phasepoints = phasepoints
        self.orders = orders
        self.ens_id = ens_id
        self.path_id = Path.path_counter
        self.connections = connections
        self.connec_string = connec_string
        self.gen = gen
        self.status = status
        self.simcycle = simcycle
        self.omin = omin
        self.omax = omax
        self.ptype = ptype
        self.plen = plen
        self.cycle_acc = cycle_acc
        self.cycle_md = cycle_md
        Path.path_counter += 1

    def copy_path(self):
        """Returns a copy of the path. Changes made to this new path will not be
        reflected in the original path.
        """
        return Path(self.phasepoints.copy(), self.orders.copy(), self.ens_id,
                    self.connections, self.connec_string,
                    self.gen, self.status, self.simcycle,
                    self.omin, self.omax, self.ptype, self.plen, self.cycle_acc,
                    self.cycle_md)

    def time_reverse(self):
        """Returns a path with the phasepoints reversed in time and the 
        velocities negated. The order parameters are also reversed.
        """
        self.phasepoints = [(x[0], -1. * x[1]) for x in self.phasepoints[::-1]]
        self.orders.reverse()
        if self.ptype is not None:
            self.ptype = self.ptype[::-1]

    # def get_connectivity_string(self):
    #     """Returns a string representation of the connectivity of the path.
    #     """
    #     con = ""
    #     for c in self.connections:
    #         con += str(c[0]) + "," + c[2] + ";"
    #     return con[:-1]
    
    def set_connectivity_string(self, connections):
        """Sets the connections of the path from a string representation of the
        connectivity.
        """
        con = ""
        for c in connections:
            con += str(c[0]) + "," + c[2] + ";"
        self.connec_string = con[:-1]
    
    def extend_connectivity_string(self, connections):
        """Extends the connections of the path from a string representation of the
        connectivity.
        """
        con = self.connec_string
        if connections is None:
            return
        con += ";"
        for c in connections:
            con += str(c[0]) + "," + c[2] + ";"
        self.connec_string = con[:-1]

    def extract_from_connectivity_string(self, start=None, stop=None):
        """Extracts the connections of the path from a string representation of the
        connectivity.
        """
        if start is None and stop is None:
            return self.connec_string
        if start is None:
            start = 0
        if stop is None:
            stop = len(self.connec_string)
        con = (self.connec_string)
        con = con.split(";")
        #assert stop + 1 <= len(con), f"stop index {stop} is out of bounds"
        con = con[start:stop+1]
        newcon = ""
        for c in con:
            newcon += c + ";"
        return newcon[:-1]
    
    def get_connectivity_string_length(self):
        """Returns the number of connections stored in the path.
        """
        return len(self.connec_string.split(";"))
    
    def get_state_connectivity(self, flip=False):
        """Returns the state connectivity of the path.
        """
        string = ""
        assert flip in [True, False], "flip must be a boolean"
        flipid = -1 if flip else 1
        for c in self.connections[::flipid]:
            string += str(c[0][0]) + "-" + str(c[2])[::flipid] + ";"
        return string[:-1]

    # make a printable string
    def __str__(self):
        # literally print everything
        msg = "Path ID: {}\nEnsemble ID: {}\nPhasepoints: {}\nOrders: {}\n"
        msg += "Connections: {}\n" 
        msg += "Connec_string: {}\nGeneration: {}\nStatus: {}\n"
        msg += "Simulation cycle: {}\nOrder min: {}\nOrder max: {}\n"
        msg += "Path type: {}\nPath length: {}\nCycle accepted: {}\n"
        msg += "Cycle made: {}\n"
        return msg.format(self.path_id, self.ens_id, self.phasepoints[:5],
                          self.orders[:5], self.connections,
                          self.connec_string, self.gen, self.status,
                          self.simcycle, self.omin, self.omax, self.ptype,
                          self.plen, self.cycle_acc, self.cycle_md)
    


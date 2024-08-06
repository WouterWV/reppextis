import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from numba import njit
import subprocess
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def check_position(op, L, R):
    """ Checks whether a phasepoint is:
    - in the interval [L, R] : M
    - left of L              : L
    - right of R             : R

    Parameters
    ----------
    op : list of floats
        op to check
    L : float
        Left boundary of the interval
    R : float
        Right boundary of the interval

    Returns
    -------
    str
        String representing the condition of the phasepoint

    """
    return "M" if L <= op[0] <= R else "L" if op[0] < L else "R"

def plot_paths(paths, intfs=None, ax=None, start_ids=0, **kwargs):
    """ Plots the paths in the list paths, with optional interfaces intfs.
    
    Parameters
    ----------
    paths : list of :py:class:`Path` objects
        Paths to plot
    intfs : list of floats, optional
        Interfaces to plot
    ax : matplotlib axis object, optional
        Axis on which to plot the paths, by default None
    start_ids : int or list of ints, optional
        Starting indices for the paths, by default 0
    kwargs : dict
        Additional keyword arguments for the plot function

    """
    if start_ids == 0:
        start_ids = [0 for _ in paths]
    elif start_ids == "staggered":
        start_ids = [0]
        for path in paths[:-1]:
            start_ids.append(start_ids[-1] + len(path.phasepoints))
    assert len(start_ids) == len(paths)
    if ax is None:
        fig, ax = plt.subplots()
    for path, start_idx in zip(paths, start_ids):
        ax.plot([i + start_idx for i in range(len(path.phasepoints))],
                [ph[0] for ph in path.orders], "-x", **kwargs)
        # plot the first and last point again to highlight start/end phasepoints
        # it must have the same color as the line for the path
        ax.plot(start_idx, path.orders[0][0], "^",
                color=ax.lines[-1].get_color(), ms = 7)
        ax.plot(start_idx + len(path.phasepoints) - 1,
                path.orders[-1][0], "v",
                color=ax.lines[-1].get_color(), ms = 7)
    if intfs is not None:
        for intf in intfs:
            ax.axhline(intf, color="k", ls="--", lw=.5)
    if ax is None:
        fig.show()

def plot_2Dpaths(paths, intfs=None, ax=None, start_ids=0, **kwargs):
    """ Plots the paths in the list paths, with optional interfaces intfs.
    
    Parameters
    ----------
    paths : list of :py:class:`Path` objects
        Paths to plot
    intfs : list of floats, optional
        Interfaces to plot
    ax : matplotlib axis object, optional
        Axis on which to plot the paths, by default None
    start_ids : int or list of ints, optional
        Starting indices for the paths, by default 0
    kwargs : dict
        Additional keyword arguments for the plot function

    """
    if start_ids == 0:
        start_ids = [0 for _ in paths]
    elif start_ids == "staggered":
        start_ids = [0]
        for path in paths[:-1]:
            start_ids.append(start_ids[-1] + len(path.phasepoints))
    assert len(start_ids) == len(paths)
    if ax is None:
        fig, ax = plt.subplots()
    for path, start_idx in zip(paths, start_ids):
        ax.plot([ph[1] for ph in path.orders],
                [ph[0] for ph in path.orders], "-x", **kwargs)
        # plot the first and last point again to highlight start/end phasepoints
        # it must have the same color as the line for the path
        ax.plot(path.orders[0][1], path.orders[0][0], "^",
                color=ax.lines[-1].get_color(), ms = 7)
        ax.plot(path.orders[-1][1],
                path.orders[-1][0], "v",
                color=ax.lines[-1].get_color(), ms = 7)
    if intfs is not None:
        for intf in intfs:
            ax.axhline(intf, color="k", ls="--", lw=.5)
    if ax is None:
        fig.show() 

def overlay_paths(path, paths):
    """Searches for sequences of phasepoints that are identical in path and
    the list paths contained in paths. Returns a list containing a tuple for 
    each element of paths. 
    The first element of a tuple is the length of the largest sequence of 
    identical phasepoints, the second element is the index where the identical
    sequence starts in the path of paths.

    Parameters:
    -----------
    path: Path object
        Path to compare to. path.phasepoints are the phasepoints. A phasepoint
        is a tuple (x, v) of floats. We care only about the x coordinate. 
    paths: list of Path objects
        Paths to compare to path. Each path.phasepoints is a list of phasepoints
        (see above).

    Returns:
    --------
    list of tuples
        List of tuples (length of identical sequence, start index of identical
        sequence in path.phasepoints) for each path in paths.
    """
    result = []

    for p in paths:
        max_length = 0
        start_index = -1
        path_start_idx = -1  # Initialize the start index w.r.t. paths

        for i in range(len(p.phasepoints)):
            for j in range(len(path.phasepoints)):
                length = 0
                while i + length < len(p.phasepoints) and \
                        j + length < len(path.phasepoints) and \
                        p.phasepoints[i + length] == path.phasepoints[j + length]:
                    length += 1

                if length > max_length:
                    max_length = length
                    start_index = j
                    path_start_idx = i

        result.append((max_length, start_index, path_start_idx))

    return result

def remove_lines_from_file(fn, n=1):
    """Remove the last n lines from a file.
    
    Parameters
    ----------
    fn : str
        File name
    n : int, optional
        Number of lines to remove, by default 1

    """
    # We use: https://stackoverflow.com/questions/1877999/
    for i in range(n):
        with open(fn, "r+", encoding="utf-8") as f:
            # Move the pointer (similar to a cursor in a text editor) to the
            # end of the file
            f.seek(0, os.SEEK_END)
            # This code means the following code skips the very last character
            # in the file - i.e. in the case the last line is null we delete
            # the last line and the penultimate one
            pos = f.tell() - 1
            # Read each character in the file one at a time from the penultimate
            # character going backwards, searching for a newline character
            # If we find a new line, exit the search
            while pos > 0 and f.read(1) != "\n":
                pos -= 1
                f.seek(pos, os.SEEK_SET)
            # So long as we're not at the start of the file, delete all the
            # characters ahead of this position
            if pos > 0:
                f.seek(pos, os.SEEK_SET)
                f.truncate()
            # After truncating, we need to position the pointer at the end, 
            # on a new blank line..
            f.write("\n")

def get_state(snakesimul):
    """Creates a Nl x Ne array of pathdirection arrays from the current 
    list of paths in a snakeSimul object. 
    
    Parameters:
    -----------
    snakesimul: :py:class:`snakeSimulation` object
        a snakeSimulation object.
    """

    mapdic = {"LML": [-1,-1],
              "LMR": [-1, 1],
              "RML": [ 1,-1],
              "RMR": [ 1, 1],}

    # Do not include the last level, as that is the latent level
    S = np.array([[mapdic[ens.last_path.ptype] for ens in level] 
                  for level in snakesimul.ensembles[:-1]])
    
    # We don't have to skip any ensembles if there are no lambda_min interfaces
    # TODO: fix lambda_min strategy
    return S

def probe_snake_trajectory(S, e, l, sL=10):
    """ Probe a possible snake trajectory, starting from a given ensemble e
    at a given level l.
    
    Parameters
    ----------
    S : Nl x Ne x 2 array of pathstates
        S[l, e] is the pathstate at level l and ensemble e, and is one of the 
        following arrays:
        [-1,-1] (LML)
        [-1, 1] (LMR)
        [ 1,-1] (RML)
        [ 1, 1] (RMR)

    e : int
        The ensemble at which the snake trajectory starts, e \in [0, Ne-1].
    l : int
        The level at which the snake trajectory starts, l \in [0, Nl-1].
    sL : int, optional
        The maximum length of the snake trajectory, by default 10.

    Returns:
    --------
    snake : list of ensembles the snake walks through
        The snake trajectory, represented as a list of ensemble ids.
        example: [[e1, l1], [e2, l1], [e3, l4], [e2, l3], ...]
    O : Nl x Ne array of integers
        The occupation matrix. O[l, e] is the step at which the snake passes
        through ensemble e at level l. If O[l, e] = 0, the ensemble is
        not traversed.
    T : list of integers
        The time-direction list. T[i] is the direction of the i-th step of the
        snake trajectory. T[i] = 1 means the snake propagates forwards in time.
    
    Notes:
    ------
    How it works:
    1a. Define occupation matrix O = np.zeros((Nl, Ne))
    1b. Define time-direction list T = []
    1c. Set acc = False
    2. Set lc, ec, np = l, e, 1
    While True:
        3. Choose a propagation direction p \in (-1,1) and add to T
        4. Set ec = ec + S[lc, ec, [None,1,0][p]]
        5. If ec = e, set acc = True and break
        6. Choose an unoccupied level at ensemble ec, and set as lc. 
           If there is none, break
        7. Set O[lc, ec] = np and np = np + 1
    8. return acc, O, T
    """
    Nl, Ne = S.shape[0]-1, S.shape[1]
    O = np.zeros((Nl, Ne))
    propdirs = []
    snake = [[l, e]]
    lc, ec, ns = l, e, 1
    status = "ACC"
    while ns < sL:
        propdir = np.random.choice(2)
        propdirs.append(-1. if propdir == 0 else 1.)
        p = S[lc, ec, propdir]
        ec = ec + p
        free_levels = np.where(O[:,ec] == 0)[0]
        if len(free_levels) == 0:
            status = "STL"
            break
        lc = np.random.choice(free_levels)
        O[lc, ec] = ns
        snake.append([lc, ec])
        # print("Going to ensemble ", ec, " at level ", lc, " with propdir ",
        #       propdir, " and p ", p, " and S[lc, ec, p] = ", S[lc, ec, p])
        ns += 1
    return status, snake, O, propdirs

def get_swapmatrix(ensembles):
    """Creates a swap matrix for a multi-level simulation object. 
    It looks at which ensembles the last path of each ensemble is connected to. 
    For now, we only look for swaps between different ensembles. 
    
    Parameters:
    -----------
    ensembles : list of list of :py:class:`Ensemble` objects
        A list of ensembles for each level.

    Returns:
    --------
    swapmatrix : Nl*Ne x Nl*Ne array
        The swap matrix. swapmatrix[l*Ne + e, lc*Ne + ec]= 1 if ensemble e at
        level l is connected to ensemble ec at level lc. Else it is 0.
    """
    Nl = len(ensembles)
    Ne = len(ensembles[0])
    swapmatrix = np.zeros((Nl*Ne, Nl*Ne))
    for l in range(Nl):
        for e in range(Ne):
            # We are probing the connected paths at ensembles[l][e], which has
            # connections to the following ensembles
            connecs = [((connec[0][0]))\
                       for connec in ensembles[l][e].last_path.connections]
            # Remove self-connections, as we care only about extrospection.
            # Go do introspection on your own time.
            ces = [c for c in connecs if c != e]  # connected ensembles
            for ce in ces:
                swapmatrix[l*Ne + e, ce[1]*Ne + ce[0]] = 1
    return swapmatrix

def get_introspective_swap_matrix(ensembles, endpoints_only=True,
                                  binomial=False, invert=False):
    """Creates a swap matrix where all ensembles are connected to all 
    other ensembles, including themselves. Introspective paths have an 
    advantage: they can swap with themselves, and therefore infinitely.
    """
    Nl = len(ensembles)
    Ne = len(ensembles[0])
    swapmatrix = np.zeros((Nl*Ne, Ne))
    weightmatrix = np.zeros((Nl*Ne, Ne), dtype=np.float64)
    for l in range(Nl): 
        for e in range(Ne):
            # We are probing the connected paths at ensembles[l][e], which has
            # connections to the following ensembles. It's own ensemble is in 
            # here, yes.
            connecs = [((connec[0][0]))\
                       for connec in ensembles[l][e].last_path.connections]
            #print("l, e, connecs: ", l, e, connecs)
            if endpoints_only:
                connecs = [connecs[0]] + [connecs[-1]]
            for i, ce in enumerate(connecs):
                swapmatrix[l*Ne + e, ce] += 1
                if binomial:
                    assert not endpoints_only, "Binomial only works for all"
                    # i goes from 0 to len(connecs)-1 (= L-1)
                    # we only do L-1 extensions (one is given)
                    # if i = 0, we have to extend L-1 FW paths
                    # if i = L-1, we have to extend L-1 BW paths
                    # if i = 1, we have to extend 1 BW and L-2 FW
                    n, p = binom(i, len(connecs)-1)
                    weightmatrix[l*Ne + e, ce] += n
                else:
                    weightmatrix[l*Ne + e, ce] += 1
    if invert:
        # W[i,j] = 1/W[i,j] if W[i,j] > 0
        for i in range(Nl*Ne):
            for j in range(Ne):
                if weightmatrix[i, j] > 0:
                    weightmatrix[i, j] = 1/weightmatrix[i, j]
    return np.hstack([swapmatrix]*Nl), np.hstack([weightmatrix]*Nl)



def fastpermanent_repeat_prob(arr, r=None):
    """P matrix calculation for specific W matrix.
    arr: n x n array of float64
    
    mix of git/infretis and Ton Hospel stackexchange code:
    1) git.com/infretis
    2) https://codegolf.stackexchange.com/questions/97060
    --_> Works only for matrices < 36x36 or OVERFLOW

    """
    assert r is not None, "r must be an integer"
    assert arr.shape[0] < 36, "Matrix too large for fastpermanent_repeat_prob"
    assert arr.shape[0] % r == 0, "Matrix must be divisible by r"

    out = np.zeros(shape=arr.shape, dtype="float64")
    # Don't overwrite input arr
    scaled_arr = arr.copy()
    n = len(scaled_arr)
    # Rescaling the W-matrix avoids numerical instabilities when the
    # matrix is large and contains large weights from
    # high-acceptance moves
    for i in range(n):
        scaled_arr[i, :] /= np.max(scaled_arr[i, :])
    for i in range(n):
        rows = np.delete(np.arange(n), i)
        sub_arr = scaled_arr[rows, :]
        lim = n if r == 1 else n//r
        for j in range(lim):
            if scaled_arr[i][j] == 0:
                continue
            columns = np.delete(np.arange(n), j)
            M = sub_arr[:, columns]
            matrix_str = "\n".join(" ".join(map(str, row)) for row in M)
            p = subprocess.Popen(["./permanent"], stdout=subprocess.PIPE,
                                  stdin=subprocess.PIPE)
            stdout, _ = p.communicate(input=matrix_str.encode())
            stdout = stdout.decode("utf-8")
            f = int(stdout.split("=")[1].strip().split()[0])
            #f = fast_glynn_perm(M)
            out[i][j] = f * scaled_arr[i][j]
    if r == 1:
        out = out/np.max(np.sum(np.abs(out), axis=1))
    else:
        out = np.hstack([out[:,:n//r]]*r)
        out = out/np.max(np.sum(np.abs(out), axis=1))
    return out

@njit
def permanent_prob(arr):
    """P matrix calculation for specific W matrix.
    arr: n x n array of float64"""
    out = np.zeros(shape=arr.shape, dtype="float64")
    # Don't overwrite input arr
    scaled_arr = arr.copy()
    n = len(scaled_arr)
    # Rescaling the W-matrix avoids numerical instabilities when the
    # matrix is large and contains large weights from
    # high-acceptance moves
    for i in range(n):
        scaled_arr[i, :] /= np.max(scaled_arr[i, :])
    for i in range(n):
        rows = np.delete(np.arange(n), i)
        sub_arr = scaled_arr[rows, :]
        for j in range(n):
            if scaled_arr[i][j] == 0:
                continue
            columns = np.delete(np.arange(n), j)
            M = sub_arr[:, columns]
            f = fast_glynn_perm(M)
            out[i][j] = f * scaled_arr[i][j]
    return out / np.max(np.sum(out, axis=1))

@njit
def fast_glynn_perm(M):
    """Glynn permanent.
    
    M: n x n array of float64"""

    def cmp(a, b):
        if a == b:
            return 0
        elif a > b:
            return 1
        else:
            return -1

    row_comb = _sum_axis(M, axis=0)
    n = len(M)

    total = 0
    old_grey = 0
    sign = +1

    binary_power_dict = {2**i: i for i in range(n)}
    num_loops = 2 ** (n - 1)

    for bin_index in range(1, num_loops + 1):
        total += sign * _reduce_multiply(row_comb)

        new_grey = bin_index ^ (bin_index // 2)
        grey_diff = old_grey ^ new_grey
        grey_diff_index = binary_power_dict[grey_diff]
        direction = 2 * cmp(old_grey, new_grey)
        if direction:
            new_vector = M[grey_diff_index]
            row_comb += new_vector * direction

        sign = -sign
        old_grey = new_grey

    return total / num_loops

@njit
def _sum_axis(arr, axis):
    """Manual summation along the specified axis."""
    result = np.zeros(arr.shape[axis], dtype=arr.dtype)
    for i in range(arr.shape[axis]):
        result[i] = np.sum(arr[..., i])
    return result

@njit
def _reduce_multiply(arr):
    """Reduction of array elements by multiplication."""
    result = 1.0
    for elem in arr:
        result *= elem
    return result 

def select_submatrix(M, cols=None, rows=None):
    """Selects the submatrix of M at col-indices cols and row-indices rows"""
    Mt = M.copy()
    if cols is not None:
        Mt = Mt[:, cols]
    if rows is not None:
        Mt = Mt[rows, :]
    return Mt

def sample_paths(P, W, iteration=0, binomial=False, atol=0.0001):
    """Go column by column, samplin' and renormalizin'. If you assign a path
    to an ensemble, and its W[i,j] > 1, you will have a choice as that path 
    has multiple connections to the ensemble you are sampling in. In that case,
    we draw an integer [0, 1, ..., W[i,j]-1] to make a choice. 
    Actually, let's make that the default behaviour, if it's a 1, then the 
    returned draw will always be 0...
    
    Parameters:
    -----------
    P: N x N array of floats
        The probability matrix of the paths
    W: N x N array of ints
        The weights of the path. 
    """
    if iteration > 1000:
        raise ValueError("It's not working man...")
    assert P.shape == W.shape, "P and W must have the same shape"
    assert P.shape[0] == P.shape[1], "P and W must be square matrices"
    Pc, Wc = P.copy(), W.copy()
    # assert sum of each row in Pc ~= 1. tolerance 0.0001
    assert np.allclose(np.sum(Pc, axis=1), 1, atol=atol), "Pc rows not summing to 1"
    # normalize the rows of Pc
    for i in range(Pc.shape[0]):
        Pc[i] = Pc[i] / np.sum(Pc[i]) if np.sum(Pc[i]) > 0 else Pc[i]
    ids = np.ones(P.shape[0], dtype=int)*666
    choices = np.zeros(P.shape[0], dtype=int)
    done = np.zeros(P.shape[0], dtype=bool)
    N = P.shape[0]
    n = 0
    while n < P.shape[0]:
        # First we check whether there are columns or rows that have only 
        # one non-zero element. If so, we appoint that path to that ensemble.

        rows, cols = find_single_nonzero_indices(Pc)
        # print("rows: ", rows)
        # print("cols: ", cols)
        if len(rows) > 0 or len(cols) > 0:
            for row in rows: 
                # print("doing: ", row[0])
                assert not done[row[0]], "Row already done"
                idx = row[1]
                # confirm that the path is not already assigend
                if idx in ids:
                    print("We got here")
                    return sample_paths(P, W, iteration+1, binomial=binomial)
                ids[row[0]] = idx 
                # print(Pc[row[0], idx])
                if binomial:
                    choices[row[0]] = 0
                else:
                    choices[row[0]] = np.random.choice(Wc[row[0], idx])
                Pc[row[0], :] = 0
                Pc[:, idx] = 0
                done[row[0]] = True
            
            # now the columns, but skip the cols that are in rows
            for col in cols:
                if col[0] in [row[0] for row in rows]:
                    continue
                corr_row = col[0]
                if done[corr_row]:
                    return sample_paths(P, W, iteration+1, binomial=binomial)
                ids[corr_row] = col[1]
                if binomial:
                    choices[corr_row] = 0
                else:
                    choices[corr_row] = np.random.choice(Wc[corr_row, col[1]])
                Pc[corr_row, :] = 0
                Pc[:, col[1]] = 0
                done[corr_row] = True

            Pc = np.array([row / np.sum(row) if np.sum(row) > 0\
                           else row for row in Pc])
            n += len(rows) + len([col for col in cols if col not in rows])
            continue
        # choose a row that is not yet done
        row = np.random.choice(np.where(done == 0)[0])
        # check if Pc[row] is a probability array, i.e. ~1+-atol
        if not np.allclose(np.sum(Pc[row]), 1, atol=atol):
            logger.warning(f"Sum of row is not 1: {np.sum(Pc[row])}, redoing")
            print("We got here BIG TIME")
            return sample_paths(P, W, iteration+1, binomial=binomial)
        idx = np.random.choice(N, p=Pc[row])
        # choose a random integer between 0 and Wc[0, idx]-1
        if binomial:
            choice = 0
        else:
            choice = np.random.choice(Wc[row, idx])
        ids[row] = idx
        choices[row] = choice
        done[row] = True
        Pc[row, :] = 0
        Pc[:, idx] = 0
        Pc = np.array([row / np.sum(row) if np.sum(row) > 0\
                       else row for row in Pc])
        n += 1
    logger.info(f"Sampling done after {iteration} iterations")
    logger.info(f"Sampled ids: {ids}")

    # assert that all ids are unique and that all paths are assigned
    if 666 in ids or len(set(ids)) < N:
        logger.warning("Not all paths are assigned, redoing")
        return sample_paths(P, W, iteration+1, binomial=binomial)
    #assert len(set(ids)) == N, "Not all paths are assigned"
    return ids, choices

def find_single_nonzero_indices(matrix):
    rows, cols = matrix.shape
    row_indices = []
    col_indices = []
    
    for i in range(rows):
        if sum(1 for x in matrix[i, :] if x != 0) == 1:
            # add position of the non-zero element
            col_id = np.where(matrix[i, :] != 0)[0][0]
            row_indices.append((i, col_id))
            
    for j in range(cols):
        if sum(1 for x in matrix[:, j] if x != 0) == 1:
            # add position of the non-zero element
            row_id = np.where(matrix[:, j] != 0)[0][0]
            col_indices.append((row_id, j))
            
    return row_indices, col_indices

# number of possibilities to draw A k times in N draws from the set {A, B}
def binom(k, N):
    """ returns the number of possibilities and the probability of drawing
    A k times in N draws from the set {A, B}, where p(A) = p(B) = 0.5."""
    # The number of possibilities is given by the binomial coefficient
    n = np.math.factorial(N) / (np.math.factorial(k) * np.math.factorial(N-k))
    # The probability distribution is given by the binomial distribution
    p = n / 2**N
    return n, p
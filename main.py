from mails import send_email
from progressbar import printProgressBar
import concurrent.futures
from datetime import datetime
from math import ceil, floor
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import networkx as nx

LOG_1_HALF = np.log(0.5)


def createRandomGraphWithPlantedClique(N, K, with_neighbors=False):
    """
    Return matrix A and planted clique vector v of a realization of a planted random graph of size N and planted clique size K
    if with_neighbors is True then it returns also a list of list of neighbors indices for each node
    """
    edge_probability = 0.5
    v_index_choices = np.random.choice(N, K, replace=False)
    v = np.array(
        [1 if i in v_index_choices else 0 for i in range(N)], dtype=np.int8)
    A = np.zeros((N, N), dtype=np.int8)
    A_neighbors = [[] for _ in range(N)]
    for i in range(N-1):
        for j in range(i + 1, N):
            if v[i] * v[j] == 1:
                A[i, j] = 1
                A_neighbors[i].append(j)
                A_neighbors[j].append(i)
            else:
                p = np.random.uniform()
                if p < edge_probability:
                    A[i, j] = 1
                    A_neighbors[i].append(j)
                    A_neighbors[j].append(i)
    A = A + A.T
    if with_neighbors:
        print("Average degree:", np.array(
            [len(x) for x in A_neighbors]).mean())
        return A, v, A_neighbors
    return A, v


def getCountLogOneHalfAndOnes(N_count_constant, x):
    """
    Count the number of times log(1/2) comes out in the second sum of the energy

    Parameters
    N_count_constant = N * (N - 1)
    x: current estimate

    Return
    count, x_sum
    count: number of times log(1/2) comes out in the second sum of the energy
    x_sum: number of elements in the current clique
    """
    x_sum = x.sum()
    # number of pairs (i,j) with x[i]*x[j] == 1
    n_pairs_1_1 = x_sum * (x_sum - 1)
    count = 0.5 * (N_count_constant - n_pairs_1_1)
    return count, x_sum


def H(x, N, log_K_over_N, log_1_minus_K_over_N, N_count_constant):
    """
    Compute the energy of the estimate x

    Parameters
    x: estimate
    N: size of the graph
    log_K_over_N, log_1_minus_K_over_N, N_count_constant: constants for computation

    Return
    energy of the estimate x
    """
    count_log_1_half, number_ones = getCountLogOneHalfAndOnes(
        N_count_constant, x)
    second_sum = count_log_1_half * LOG_1_HALF
    first_sum = number_ones * log_K_over_N
    first_sum += (N - number_ones) * log_1_minus_K_over_N
    return -first_sum - second_sum


def drawCandidate(x, N, K, A, method="switch_standard", p=0.5, k=1, param_remove=0.5, beta=-1.0, A_neighbors=None, nodes_probabilities=[]):
    """
    Ensure that all inserted elements are linked to the other elements of the current clique
    Candidate elements (to be added) are chosen among the common neighbors of the current elements of the clique

    Parameters
    x: current estimate
    N, K, A: graphs variables
    methods:
        switch_standard: "standard" procedure
        switch_k: switch k element of x: k_remove = (param_remove * min(k, size of current clique)) elements are removed with probability p, (size of current clique - k_remove + 1) from the common neighbors of the current clique x are added if they form a clique up to a limit
        switch_k_add: same as switch_k but add (k - k_remove) elements with probability (p + 0.25 * beta) of it forms a clique
    p: (only for switch_k method) the probability of switch acceptance
    k: (only for switch_k method) the minimum number of elements considered to be switched
    param_remove: (only for switch_k method) the proportion of elements that will be tried to be removed from the current clique
    beta: inverse temperature
    A_neighbors: list of list of neighbors for each node of the graph
    nodes_probabilities: list of probabilities for each node (number of neighbors / (2 * total number of edges))

    Return
    return a candidate (np array with entries 0 or 1), and an integer representing the order of the number of operations needed to compute the common neighbors of the current clique (0 if method is switch_standard)
    """
    x_candidate = np.copy(x)
    time_complexity = 0
    if method in ["switch_standard", "switch_k", "switch_k_add", "switch_all"]:
        if method == "switch_standard":
            # standard procedure: pick one node, if in clique then remove with probability exp(-beta),
            # else if adding the node forms a clique then add it to the candidate
            if len(nodes_probabilities) == 0:
                random_index = np.random.choice(N, 1)[0]
            else:
                random_index = np.random.choice(
                    N, 1, p=nodes_probabilities)[0]
            if x_candidate[random_index] == 1:
                p_switch = np.random.uniform()
                if p_switch <= np.exp(-beta):
                    x_candidate[random_index] = 0
                return x_candidate, time_complexity
            make_a_clique = True
            for j in range(N):
                if x_candidate[j] == 0 or j == random_index:
                    continue
                if A[random_index, j] != 1:
                    make_a_clique = False
                    break
            if make_a_clique:
                x_candidate[random_index] = 1
                return x_candidate, time_complexity
            return x_candidate, time_complexity
        elif method == "switch_all":
            estimate_indices = [i for i in range(N) if x_candidate[i] == 1]
            # for i in range(N):
            #     p_switch = np.random.uniform()
            #     if p_switch < 0.5:
            #         if x_candidate[i] == 1:
            #             x_candidate[i] = 0
            #             estimate_indices.remove(i)
            #         else:
            #             make_a_clique = True
            #             for j in estimate_indices:
            #                 if A[i, j] != 1:
            #                     make_a_clique = False
            #                     break
            #             # if len([j for j in estimate_indices if A[i, j] != 1]) == 0:
            #             if make_a_clique:
            #                 x_candidate[i] = 1
            #                 estimate_indices.append(i)
            number_accepted = np.random.binomial(N, 0.5)
            switch_accepted = np.random.choice(
                N, number_accepted, replace=False)
            for i in switch_accepted:
                if x_candidate[i] == 1:
                    x_candidate[i] = 0
                    estimate_indices.remove(i)
                else:
                    make_a_clique = True
                    for j in estimate_indices:
                        if A[i, j] != 1:
                            make_a_clique = False
                            break
                    # if len([j for j in estimate_indices if A[i, j] != 1]) == 0:
                    if make_a_clique:
                        x_candidate[i] = 1
                        estimate_indices.append(i)
        else:
            add_with_probability = method == "switch_k_add"
            if N < k:
                k = N
            clique_indices1 = [i for i in range(N) if x_candidate[i] == 1]
            k_remove = max(
                1, floor(min(len(clique_indices1), k) * param_remove))
            if add_with_probability:
                k_add = k - k_remove
            else:
                k_add = len(clique_indices1) - k_remove + 1
            if len(clique_indices1) == 0:
                # if the current clique is empty, then try to add a quarter of the target size
                k_add = floor(K * 0.25)
            else:
                # remove k_remove elements of the clique with probability p
                if k_remove > len(clique_indices1):
                    k_remove = len(clique_indices1)
                    if add_with_probability:
                        k_add = k - k_remove
                    else:
                        k_add = len(clique_indices1) - k_remove + 1
                choice_remove = np.random.choice(
                    clique_indices1, k_remove, replace=False)
                for i in choice_remove:
                    p_switch = np.random.uniform()
                    if p_switch < p:
                        x_candidate[i] = 0

            common_neighbors = []
            # the difference between the 2 statements below to compute the common neighbors is only for time purpose
            if A_neighbors == None:
                for i in range(N):
                    if x_candidate[i] == 0:
                        continue
                    if len(common_neighbors) == 0:
                        common_neighbors = [
                            j for j in range(N) if A[i, j] == 1]
                        continue
                    common_neighbors = [
                        j for j in common_neighbors if A[i, j] == 1]
                if len(common_neighbors) == 0:
                    if x_candidate.sum() == 0:
                        common_neighbors = [i for i in range(N)]
            else:
                if x_candidate.sum() == 0:
                    common_neighbors = [i for i in range(N)]
                else:
                    rand_indices = np.random.choice(N, N, replace=False)
                    for i in rand_indices:
                        if x_candidate[i] == 0:
                            continue
                        if len(common_neighbors) == 0:
                            common_neighbors = A_neighbors[i]
                        else:
                            time_complexity += min(len(common_neighbors),
                                                   len(A_neighbors[i]))
                            new_common_neighbors = list(
                                set(common_neighbors).intersection(A_neighbors[i]))
                            if len(new_common_neighbors) == 0:
                                # if there are no common neighbor for the elements of the clique, remove this element from the clique
                                x_candidate[i] = 0
                                k_add -= 1
                            else:
                                common_neighbors = new_common_neighbors
                            # time complexity is O(min(len(common_neighbors), len(A_neighbors[i])))
            size_of_the_clique_before_adding = x_candidate.sum()
            if k_add > len(common_neighbors):
                k_add = len(common_neighbors)
            if len(nodes_probabilities) == 0:
                if add_with_probability:
                    choice_add = np.random.choice(
                        common_neighbors, k_add, replace=False)
                else:
                    choice_add = np.random.choice(
                        common_neighbors, len(common_neighbors), replace=False)
            else:
                nodes_probabilities_add = [
                    nodes_probabilities[i] for i in common_neighbors]
                nodes_probabilities_sum = np.array(
                    nodes_probabilities_add).sum()
                nodes_probabilities_add = [
                    x / nodes_probabilities_sum for x in nodes_probabilities_add]
                nodes_probabilities_add[-1] = 1.0 - \
                    np.array(nodes_probabilities_add[:-1]).sum()
                if add_with_probability:
                    choice_add = np.random.choice(
                        common_neighbors, k_add, replace=False, p=nodes_probabilities_add)
                else:
                    choice_add = np.random.choice(
                        common_neighbors, len(common_neighbors), replace=False, p=nodes_probabilities_add)
            # the nodes added to the candidate
            added = []
            if not add_with_probability:
                limit_add = max(
                    1, min(K - size_of_the_clique_before_adding, floor((K - size_of_the_clique_before_adding) * 0.5) + 1))
                if len(clique_indices1) == 0:
                    limit_add = k_add + 1
            # add nodes in common neighbors of the current clique to the candidate according to the method
            for i in choice_add:
                if add_with_probability:
                    p_switch = np.random.uniform()
                    p_accept = p
                    if beta > 0:
                        p_accept += beta * 0.25
                    if p_switch <= p_accept:
                        if size_of_the_clique_before_adding + len(added) < K and len([j for j in added if A[i, j] != 1]) == 0:
                            x_candidate[i] = 1
                            added.append(i)
                else:
                    if len(added) < limit_add:
                        if len([j for j in added if A[i, j] != 1]) == 0:
                            x_candidate[i] = 1
                            added.append(i)
            return x_candidate, time_complexity
    return x_candidate, time_complexity


def metropolisHastings(A, N, K, x_init, n_steps, log_K_over_N, log_1_minus_K_over_N, N_count_constant, beta=1.0, print_progress=False, A_neighbors=None, with_time_complexity=False, nodes_probabilities=[], draw_method=""):
    """
    Perform n_steps of the Metropolis Hastings algorithm

    Parameters:
    A, N, K, x_init, n_steps: in the context are self explanatory
    log_K_over_N, log_1_minus_K_over_N, N_count_constant: are constants for computation
    beta: the inverse temperature
    print_progress: bool, show or not a progress bar (when used in the PT algorithm, please set to False)
    A_neighbors: see createRandomGraphWithPlantedClique
    with_time_complexity: whether to compute the order of the operations needed to compute the common neighbors in the drawCandidate method
    nodes_probabilities: list of probabilities for each node (number of neighbors / (2 * total number of edges))

    Return:
    x, H_x, info
    x: np array representing the new estimate
    H_x: float, energy associated with the new estimate
    count_changes: # of time the candidate has been accepted
    if with_time_complexity is True: return also the order of the # of operations needed in the drawCandidate method
    """
    # the method used in the drawCandidate function
    candidate_method = "switch_all"  # "switch_k"
    if len(draw_method) > 0:
        candidate_method = draw_method
    param_k = getParam_k(N, K)
    param_remove = getParam_Remove(N, K)
    x = np.copy(x_init)
    H_x = H(x, N, log_K_over_N, log_1_minus_K_over_N, N_count_constant)
    count_changes = 0
    size_of_clique = 0
    count_equal = 0
    if print_progress:
        printProgressBar(0, n_steps, prefix=f"Progress:",
                         suffix="Complete (size of clique estimate: 0)", length=20)
    time_complexity = 0
    for i in range(n_steps):
        if print_progress:
            printProgressBar(i + 1, n_steps, prefix=f"Progress:",
                             suffix=f"Complete (size of clique estimate: {size_of_clique})", length=20)
        p = 0.5
        x_candidate, step_time_complexity = drawCandidate(
            x, N, K, A, candidate_method, p=p, k=param_k, param_remove=param_remove, beta=beta, A_neighbors=A_neighbors, nodes_probabilities=nodes_probabilities)
        H_candidate = H(x_candidate, N,
                        log_K_over_N, log_1_minus_K_over_N, N_count_constant)
        time_complexity += step_time_complexity
        if H_candidate == float("inf"):
            # this should not happen
            continue
        elif H_x == float("inf"):
            # this should not happen
            count_changes += 1
            x = x_candidate
            H_x = H_candidate
            size_of_clique = x.sum()
            continue
        if H_candidate <= H_x:  # alpha >= 1.0:
            if H_candidate < H_x:
                count_changes += 1
            else:
                if len([i for i in range(N) if x[i] == 1 and x_candidate[i] != 1]) != 0 or len([i for i in range(N) if x_candidate[i] == 1 and x[i] != 1]) != 0:
                    count_changes += 1
                else:
                    count_equal += 1
            x = x_candidate
            H_x = H_candidate
            size_of_clique = x.sum()
        else:
            alpha = np.exp(-beta * (H_candidate - H_x))
            p_accept = np.random.uniform()
            if p_accept < alpha:
                count_changes += 1
                x = x_candidate
                H_x = H_candidate
                size_of_clique = x.sum()
    if count_equal < n_steps:
        count_changes = float(count_changes) / (n_steps - count_equal)
    else:
        count_changes = -1
    if with_time_complexity:
        return x, H_x, count_changes, time_complexity
    return x, H_x, count_changes


def performMetropolisOnAllReplicas(estimates, betas, A, N, K, n_steps, log_K_over_N, log_1_minus_K_over_N, N_count_constant, with_threading=False, A_neighbors=None, nodes_probabilities=[], draw_method=""):
    """
    Call the Metropolis algorithm for each replica

    Parameters: self explanatory in the context
    with_threading: whether to perform the Metropolis calls in parallel for each beta or not

    Return:
    new_estimates: list of np arrays representing the new estimates for each replica
    new_energies: list of float of the energies associated to the new estimates
    monitoring: list of info (see metropolisHastings) for each replica
    avg_time_complexity: order of the average # of operations needed in the drawCandidate calls (averaged over the different betas)
    """
    new_estimates = [np.zeros(N) for _ in range(len(betas))]
    new_energies = [0 for _ in range(len(betas))]
    monitoring = [0 for _ in range(len(betas))]
    time_complexities = [0 for _ in range(len(betas))]
    if with_threading:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(metropolisHastings, A, N, K, estimates[i], n_steps, log_K_over_N, log_1_minus_K_over_N,
                                       N_count_constant, betas[i], A_neighbors=A_neighbors, with_time_complexity=True, nodes_probabilities=nodes_probabilities, draw_method=draw_method) for i in range(len(betas))]
        for i, f in enumerate(futures):
            new_estimates[i], new_energies[i], monitoring[i], time_complexities[i] = f.result(
            )
    else:
        for i, beta in enumerate(betas):
            x, H_x, count_changes, time_complexity = metropolisHastings(
                A, N, K, estimates[i], n_steps, log_K_over_N, log_1_minus_K_over_N, N_count_constant, beta, A_neighbors=A_neighbors, with_time_complexity=True, nodes_probabilities=nodes_probabilities, draw_method=draw_method)
            new_estimates[i] = x
            new_energies[i] = H_x
            monitoring[i] = count_changes
            time_complexities[i] = time_complexity
    avg_time_complexity = np.array(time_complexities).mean()
    return new_estimates, new_energies, monitoring, avg_time_complexity


def performSwitchConfiguration(estimates, energies, betas, config={"how": "byPairs", "reverse": False}):
    """
    Perform the switch of configurations according to current energies

    Parameters:
        estimates: array of np arrays with estimates for each inverse temperatures beta
        energies: array of float with energies associated with the current estimate for each inverse temperatures beta
        betas: array of float representing the inverse temperatures beta
        config: dict with keys "how": str (either "byPairs": the switch is tried between distinct pairs (at indices 0 and 1, then at indices 2 and 3, etc.) or anything else: try to switch with the consecutive estimate for each beta), "reverse": whether it is run from the estimate at index 0 to end or from end to estimate with index 0 (ignored if "how" is "byPairs")

    Return: new array of estimates, new array of energies, monitoring (dict with keys "switchCount": how many switch of configurations happened, "switchBeta1": how many switch of configurations happened involving estimate at index 0)
    """
    new_estimates = [np.copy(estimates[i]) for i in range(len(betas))]
    new_energies = [energies[i] for i in range(len(betas))]
    monitoring = {"switchCount": 0, "switchBeta1": 0}
    param_pair_step = 2 if config["how"] == "byPairs" else 1
    param_start = 0
    param_stop = len(betas) - 1
    if config["how"] != "byPairs" and "reverse" in config and config["reverse"]:
        param_start = len(betas) - 2
        param_stop = 0
        param_pair_step = -1
    for i in range(param_start, param_stop, param_pair_step):
        p_switch = np.exp((betas[i] - betas[i + 1]) *
                          (new_energies[i] - new_energies[i + 1]))
        # p_switch = min(1, p_switch) # no need to do that in practice
        p = np.random.uniform()
        if p <= p_switch:
            monitoring["switchCount"] += 1
            if i == 0 or i + 1 == 0:
                monitoring["switchBeta1"] += 1
            new_estimates_i = np.copy(new_estimates[i])
            new_estimates[i] = np.copy(new_estimates[i + 1])
            new_estimates[i + 1] = new_estimates_i
            new_energies_i = new_energies[i]
            new_energies[i] = new_energies[i + 1]
            new_energies[i + 1] = new_energies_i
    if config["how"] == "byPairs" and len(betas) > 1:
        for i in range(1, len(betas) - 2 - (1 - len(betas) % 2), 2):
            p_switch = np.exp((betas[i] - betas[i + 1]) *
                              (new_energies[i] - new_energies[i + 1]))
            # p_switch = min(1, p_switch) # no need to do that in practice
            p = np.random.uniform()
            if p <= p_switch:
                monitoring["switchCount"] += 1
                if i == 0 or i + 1 == 0:
                    monitoring["switchBeta1"] += 1
                new_estimates_i = np.copy(new_estimates[i])
                new_estimates[i] = np.copy(new_estimates[i + 1])
                new_estimates[i + 1] = new_estimates_i
                new_energies_i = new_energies[i]
                new_energies[i] = new_energies[i + 1]
                new_energies[i + 1] = new_energies_i
    return new_estimates, new_energies, monitoring


def get_coordinates_in_circle(n):
    """
    Return list of coordinates to draw a graph with the planted clique in a circle
    """
    thetas = [2*np.pi*(float(i)/n) for i in range(n)]
    returnlist = [(np.cos(theta), np.sin(theta)) for theta in thetas]
    return returnlist


def parallelTempering(A, N, K, betas, n_steps=5, switchConfig={"how": "consecutive", "reverse": False}, A_neighbors=None, with_threading=True, nodes_probabilities=[], show_graph=False, v_indices=[], init_near_solution=False):
    """
    Perform the parallel tempering method with Metropolis Hastings steps

    Parameters:
    A: np matrix according to the paper (with variables A_ij = 1 if v[i]*v[j] == 1, else: 0 or 1 with probability 1/2 each)
    N: size of the graph
    K: size of the planted clique
    betas: list of inverse temperatures
    n_steps: number of Metropolis steps for each replica before each configuration switch try
    switchConfig: see performSwitchConfiguration
    A_neighbors: list of list of neighbors for each node of the graph
    with_threading: whether to perform the Metropolis steps in parallel for each replica or not
    nodes_probabilities: list of probabilities for each node (number of neighbors / (2 * total number of edges))
    show_graph: whether to create a visual representation of the evolution of the search (it requires v_indices)
    v_indices: the indices of the nodes of the planted clique (only for init_near_solution or show_graph options)
    init_near_solution: whether to init the estimate near the solution or not (if True initialize the estimate with 0.33 * K correct nodes)

    Return:
    x, monitoring_metropolis, monitoring_tempering, iterations
    x: the estimated clique
    monitoring_metropolis, monitoring_tempering: monitoring
    iterations: {"iterations": number of iterations done, "time": time needed, "avgTimeComplexity": average time complexity for the drawCandidate method}
    """
    start = datetime.now()

    log_K_over_N = np.log(K / N)  # constant for computation
    log_1_minus_K_over_N = np.log(1 - K / N)  # constant for computation
    N_count_constant = N * (N - 1)  # constant for computation
    betas_index_middle = floor(len(betas) * 0.5)  # used for monitoring

    control = 0  # current number of iteration of the algorithm
    # maximum number of iterations of the algorithm
    limit = 2000000  # prevent infinite search

    # initialization of the estimates for each replica
    estimates = [np.zeros(N) for _ in range(len(betas))]
    if init_near_solution:
        # initialize the estimate with 0.33 * K correct nodes
        if len(v_indices) != K:
            print("Please provide the parameter v_indices to initialize near solution")
            return estimates[0], monitoring_metropolis, {"switchCount": 0, "switchCountBeta1": 0}, {"iterations": control, "time": 0, "avgTimeComplexity": 0}
        elements = np.random.choice(
            v_indices, max(1, floor(0.33 * K)), replace=False)
        for i in elements:
            for j in range(len(estimates)):
                estimates[j][i] = 1

    # initialization of the energies of the estimates for each replica
    energies = [0.0 for _ in range(len(betas))]

    # keep track of metropolis changes acceptance for each replica
    monitoring_metropolis = [0 for _ in range(len(betas))]

    # keep track of the current total number of switches of configurations
    current_number_changes_temp = 0
    current_number_changes_temp_beta_1 = 0

    size_estimate_clique = 0  # current estimated clique size

    avg_time_complexity = 0

    # threshold = 1.05 * np.log2(N) / K

    if show_graph:
        # create a plot with visual representations of the search in the graph for each replica
        # the plots will be saved in the folder "plots/"
        figure, axis = plt.subplots(2, ceil(len(betas) * 0.5))
        figure.set_size_inches(
            (6.4 + 0.8) * ceil(len(betas) * 0.5), (4.8 + 0.8) * 2)
        if len(betas) % 2 != 0:
            for i in range(2 * ceil(len(betas) * 0.5) - len(betas)):
                figure.delaxes(axis[1, ceil(len(betas) * 0.5) - (i + 1)])
        G = nx.from_numpy_matrix(A)
        circular_positions = get_coordinates_in_circle(K)
        pos = {}
        for i, p in enumerate(v_indices):
            pos[p] = circular_positions[i]
        pos = nx.spring_layout(
            G, pos=pos, fixed=v_indices, seed=3113794652, k=5.0/np.sqrt(N))
        d = nx.degree(G)
        d = [(d[node]+1) * 2 for node in G.nodes()]
        for i, b in enumerate(betas):
            row, col = 0, i
            if i >= ceil(len(betas) * 0.5):
                row = 1
                col = i - ceil(len(betas) * 0.5)
            nx.draw(G, pos=pos, node_color=[
                "tab:blue" if i in v_indices else "tab:gray" for i in range(N)], node_size=d, ax=axis[row, col])
            axis[row, col].set_title(f"beta={round(b, 2)}")
        plt.savefig(f"plots/subplot_N{N}_K{K}_0.png", dpi=300)

    # initialize the progress bar indicating the percentage of the current estimated clique size against K
    printProgressBar(0, K, prefix=f"Progress:",
                     suffix=f"of the clique size (step #{control}, beta config flips: {current_number_changes_temp})", length=20)
    # run the algorithm
    while control < limit and size_estimate_clique < K:
        # perform Metropolis on all replicas
        draw_method = "switch_all"
        if control % 2 == 0 and float(size_estimate_clique) / K > 0.58:
            draw_method = "switch_k"
        estimates, energies, new_monitoring_metropolis, step_avg_time_complexity = performMetropolisOnAllReplicas(
            estimates, betas, A, N, K, n_steps, log_K_over_N, log_1_minus_K_over_N, N_count_constant, A_neighbors=A_neighbors, with_threading=with_threading, nodes_probabilities=nodes_probabilities, draw_method=draw_method)
        # monitor changes by Metropolis
        monitoring_metropolis = [(control * monitoring_metropolis[i] + (new_monitoring_metropolis[i] if new_monitoring_metropolis[i] >= 0 else monitoring_metropolis[i])) / (
            control + 1) for i in range(len(monitoring_metropolis))]

        avg_time_complexity += step_avg_time_complexity

        if show_graph:
            # visualization of the search before switching the configurations
            for j, b in enumerate(estimates):
                row, col = 0, j
                if j >= ceil(len(betas) * 0.5):
                    row = 1
                    col = j - ceil(len(betas) * 0.5)
                nx.draw(G, pos=pos, node_color=[
                    "tab:green" if i in v_indices and estimates[j][i] == 1 else "tab:blue" if i in v_indices else "tab:red" if estimates[j][i] == 1 else "tab:gray" for i in range(N)], node_size=d, ax=axis[row, col])
            plt.savefig(
                f"plots/subplot_N{N}_K{K}_{control + 1}.png", dpi=300)

        # perform configurations
        estimates, energies, monitoring_tempering_step = performSwitchConfiguration(
            estimates, energies, betas, switchConfig)

        # keep track of the configurations switches
        current_number_changes_temp += monitoring_tempering_step["switchCount"]
        current_number_changes_temp_beta_1 += monitoring_tempering_step["switchBeta1"]

        # size of the current estimated clique
        size_estimate_clique = estimates[0].sum()

        if show_graph:
            # visualization of the search after switching the configurations
            for j, b in enumerate(estimates):
                row, col = 0, j
                if j >= ceil(len(betas) * 0.5):
                    row = 1
                    col = j - ceil(len(betas) * 0.5)
                nx.draw(G, pos=pos, node_color=[
                    "tab:green" if i in v_indices and estimates[j][i] == 1 else "tab:blue" if i in v_indices else "tab:red" if estimates[j][i] == 1 else "tab:gray" for i in range(N)], node_size=d, ax=axis[row, col])
            plt.savefig(
                f"plots/subplot_N{N}_K{K}_{control + 1}_afterswitch.png", dpi=300)

        # update progress bar
        change_accept_beta_1 = round(
            monitoring_metropolis[0] * 100, 1)
        change_accept_beta_0_55 = round(
            monitoring_metropolis[betas_index_middle] * 100, 1)
        change_accept_beta_0_1 = round(
            monitoring_metropolis[-1] * 100, 1)
        if init_near_solution or len(v_indices) > 0:
            printProgressBar(size_estimate_clique, K, prefix=f"Progress:",
                             suffix=f"of the clique size (step #{control + 1}, {len([i for i in v_indices if estimates[0][i] == 1])} - {len([i for i in v_indices if estimates[betas_index_middle][i] == 1])} - {len([i for i in v_indices if estimates[-1][i] == 1])} of solution, beta config flips: {current_number_changes_temp}, accept: | 1.0: {change_accept_beta_1}%, 0.55: {change_accept_beta_0_55}%, 0.1: {change_accept_beta_0_1}%)", length=20)
        else:
            printProgressBar(size_estimate_clique, K, prefix=f"Progress:",
                             suffix=f"of the clique size (step #{control + 1}, beta config flips: {current_number_changes_temp}, accept: | 1.0: {change_accept_beta_1}%, {round(betas[betas_index_middle], 2)}: {change_accept_beta_0_55}%, {round(betas[-1], 2)}: {change_accept_beta_0_1}%)", length=20)
        # an iteration was done
        control += 1

    avg_time_complexity = (avg_time_complexity * len(betas)) / control

    # the clique has not been recovered inside the limit
    if size_estimate_clique != K:
        printProgressBar(K, K, prefix=f"Progress:",
                         suffix=f"of the clique size (step #{control + 1}, beta config flips: {current_number_changes_temp})", length=20)
        print("Failed to recover")

    stop = datetime.now()

    return estimates[0], monitoring_metropolis, {"switchCount": current_number_changes_temp, "switchCountBeta1": current_number_changes_temp_beta_1}, {"iterations": control, "time": (stop - start).seconds, "avgTimeComplexity": avg_time_complexity}


def testParallelTempering(N, K, betas=[], n_steps=5, show_graph=False):
    """
    Test the parallel tempering algorithm
    """

    # print test settings
    print("===================== START test =====================")
    print("SETTINGS:")
    print("N", N)
    print("K", K)

    # create planted random graph
    A, v, A_neighbors = createRandomGraphWithPlantedClique(
        N, K, with_neighbors=True)
    truth = [i for i in range(N) if v[i] == 1]

    # initialize inverse temperatures
    if len(betas) == 0:
        betas = [1 - i * 0.15 for i in range(7)]

    # run PT and compute elapsed time
    v_indices = []
    if show_graph:
        v_indices = truth
    estimate, monitoring_metropolis, monitoring_tempering, time_result = parallelTempering(
        A, N, K, betas, n_steps, A_neighbors=A_neighbors, with_threading=True, show_graph=show_graph, v_indices=v_indices)

    # result compared to actual clique
    estimate_indices = [i for i in range(N) if estimate[i] == 1]
    diff_not_in_truth = [i for i in estimate_indices if i not in truth]
    diff_not_in_estimate = [i for i in truth if i not in estimate_indices]
    try:
        print("Clique size in estimate:", len(estimate_indices))
        print("Count not in truth but in estimate:", len(diff_not_in_truth))
        print("Count not in estimate but in truth:", len(diff_not_in_estimate))
        print("RESULT (recovered?):", len(diff_not_in_truth)
              == 0 and len(diff_not_in_estimate) == 0)
    except:
        pass

    # time needed
    try:
        print("Total # of iterations:", time_result["iterations"])
        time_needed = time_result["time"]
        print("Elapsed time:", floor(time_needed / 60.0), "min",
              time_needed - floor(time_needed / 60.0) * 60, "sec")
    except:
        pass

    # tempering monitoring
    try:
        tempering_switch_total_count = monitoring_tempering["switchCount"]
        tempering_switch_beta_1_count = monitoring_tempering["switchCountBeta1"]
        print("Total tempering switch:", tempering_switch_total_count)
        print("Total tempering switch for beta = 1.0:",
              tempering_switch_beta_1_count)
    except:
        pass

    # metropolis monitoring
    try:
        print("Changes for beta = 1.0 (%):",
              monitoring_metropolis[0] * 100)
    except:
        pass
    print("====================== END test ======================")


def getKFromKTilde(N, K_tilde):
    """
    Return the size of the clique for a given K_tilde
    """
    return max(1, round(K_tilde * np.log2(N)))


def getParam_k(N, K):
    """
    Return the param_k for the drawCandidate method
    """
    if K > np.sqrt(N / np.e):
        return min(60, max(1, floor(K * 0.5))) + floor(N / 1000.0)
    return max(1, 2 * K)


def getParam_Remove(N, K):
    """
    Return the param_remove for the drawCandidate method
    """
    if K > np.sqrt(N / np.e):
        return 0.25
    return 0.5


def checkIfClique(x, A):
    """
    Check whether an estimate is a clique or not
    """
    for i in range(len(x)):
        if x[i] == 1:
            for j in range(len(x)):
                if j != i and x[j] == 1 and A[i, j] != 1:
                    return False
    return True


def timeOfConvergenceChangingN(Ns, n_samples, K_as_list=[], K_tilde=-1, K_to_N_factor=0.125, send_result_email=False):
    """
    Run the PT algorithm for each N in Ns (n_samples for each N) and save # of iterations needed for the PT in files

    Parameters:
    Ns: list of integers (sizes of graphs to be sampled)
    n_samples: the number of graph's realizations for each N in Ns
    K_to_N_factor: if K_tilde < 0 and K_as_list = [] then the size of the planted clique is N * K_to_N_factor
    K_tilde: if K_as_list = [] then compute the size of the planted clique as K_tilde * log_2(N) 
    K_as_list: if non empty then compute the size of the planted clique as K_as_list[i] for i in range(len(Ns))
    """
    pwd = ""
    if send_result_email:
        pwd = input("Input password and press Enter: ")
    results = [[] for _ in range(len(
        Ns))]  # save the results (number of PT iterations needed to recover the planted clique)
    betas = [1 - i * 0.15 for i in range(7)]  # default betas
    n_steps = 5  # Metropolis steps
    for i, N in enumerate(Ns):
        avg_time_complexity = 0
        if len(K_as_list) > 0:
            K = K_as_list[i]
        elif K_tilde > 0:
            K = getKFromKTilde(N, K_tilde)
        else:
            K = floor(K_to_N_factor * N)
        print("===================== START sampling =====================")
        print("SETTINGS:")
        print("N", N)
        print("K", K)
        samples_done_count = 0
        while samples_done_count < n_samples:
            A, v, A_neighbors = createRandomGraphWithPlantedClique(
                N, K, with_neighbors=True)
            truth = [i for i in range(N) if v[i] == 1]
            nodes_probabilities = [
                len(A_neighbors[i]) + max(0, len(A_neighbors[i]) - N * 0.5) for i in range(N)]
            total = np.array(nodes_probabilities).sum()
            nodes_probabilities = [
                float(x) / total for x in nodes_probabilities]
            nodes_probabilities[-1] = 1.0 - \
                np.array(nodes_probabilities[:-1]).sum()
            estimate, monitoring_metropolis, monitoring_tempering, time_res = parallelTempering(
                A, N, K, betas, n_steps, A_neighbors=A_neighbors, nodes_probabilities=nodes_probabilities)
            estimate_indices = [i for i in range(N) if estimate[i] == 1]
            diff_not_in_truth = [
                i for i in estimate_indices if i not in truth]
            diff_not_in_estimate = [
                i for i in truth if i not in estimate_indices]
            if len(diff_not_in_truth) == 0 and len(diff_not_in_estimate) == 0:
                results[i].append(time_res["iterations"])
                samples_done_count += 1
                avg_time_complexity += time_res["avgTimeComplexity"]
                print(
                    f"Clique {samples_done_count} recovered (N: {N}, K: {K}, TC: {round(avg_time_complexity / samples_done_count)})")
            else:
                estimate_is_clique = checkIfClique(estimate, A)
                print("WARNING: planted clique not recovered")
                print("Estimate is clique:",
                      estimate_is_clique)
                print("Estimate has size K",
                      len(estimate_indices) == K)
                if estimate_is_clique:
                    common_elements = float(
                        len([i for i in estimate_indices if i in truth])) / K
                    print("Estimate common elements with planted clique:",
                          round(common_elements * 100, 1), "%")
        print(f"Sampling for N={N} finished with time complexity:",
              round(avg_time_complexity / samples_done_count))
    filename_suffix = datetime.now().isoformat()
    filename_suffix = filename_suffix.replace(":", "-")
    if "." in filename_suffix:
        filename_suffix = filename_suffix[:filename_suffix.index(".")]
        filename_suffix = filename_suffix.replace(".", "-")
    with open(f"final_results_time_of_convergence_changing_N_from_{Ns[0]}_to_{Ns[-1]}_{n_samples}samples_{filename_suffix}.npy", "wb") as f:
        np.save(f, np.array(results))
        np.save(f, np.array(Ns))
        if len(K_as_list) > 0:
            np.save(f, np.array(K_as_list))
        elif K_tilde > 0:
            np.save(f, np.array([K_tilde, 0]))
        else:
            np.save(f, np.array([0, K_to_N_factor]))
    print(
        "Saved:", f"final_results_time_of_convergence_changing_N_from_{Ns[0]}_to_{Ns[-1]}_{n_samples}samples_{filename_suffix}.npy")
    if send_result_email:
        status = send_email("yan-lhomme@outlook.com", "Results time of convergence",
                            f"final_results_time_of_convergence_changing_N_from_{Ns[0]}_to_{Ns[-1]}_{n_samples}samples_{filename_suffix}.npy", pwd)
        print("Sent email:", status)


def timeOfConvergenceChangingK(Ns, K_tildes, n_samples=1, accept_other_clique=False, send_result_email=False):
    """
    Run the PT algorithm for each N in Ns and over the different K_tilde in the corresponding list

    Parameters:
    Ns: list of integers (sizes of graphs to be sampled)
    K_tildes: list of list of K_tilde values
    n_samples: number of samples for each K_tilde
    """
    pwd = ""
    if send_result_email:
        pwd = input("Input password and press Enter: ")
    results = []
    betas = [1 - i * 0.15 for i in range(7)]
    n_steps = 5  # Metropolis steps
    for i, N in enumerate(Ns):
        results.append([])
        for j, K_tilde in enumerate(K_tildes[i]):
            results[i].append([])
            K = getKFromKTilde(N, K_tilde)
            print("===================== START sampling =====================")
            print("SETTINGS:")
            print("N", N)
            print("K", K)
            print("K_tilde", K_tilde)
            realizations_done_count = 0
            while realizations_done_count < n_samples:
                A, v, A_neighbors = createRandomGraphWithPlantedClique(
                    N, K, with_neighbors=True)
                truth = [i for i in range(N) if v[i] == 1]
                nodes_probabilities = [
                    len(A_neighbors[i]) + max(0, len(A_neighbors[i]) - N * 0.5) for i in range(N)]
                total = np.array(nodes_probabilities).sum()
                nodes_probabilities = [
                    float(x) / total for x in nodes_probabilities]
                nodes_probabilities[-1] = 1.0 - \
                    np.array(nodes_probabilities[:-1]).sum()
                estimate, monitoring_metropolis, monitoring_tempering, time_res = parallelTempering(
                    A, N, K, betas, n_steps, A_neighbors=A_neighbors, nodes_probabilities=nodes_probabilities)
                estimate_indices = [i for i in range(N) if estimate[i] == 1]
                diff_not_in_truth = [
                    i for i in estimate_indices if i not in truth]
                diff_not_in_estimate = [
                    i for i in truth if i not in estimate_indices]
                if len(diff_not_in_truth) == 0 and len(diff_not_in_estimate) == 0:
                    results[i][j].append(time_res["iterations"])
                    realizations_done_count += 1
                    print(
                        f"Clique {realizations_done_count} recovered (N: {N}, K_tilde: {K_tilde})")
                else:
                    if accept_other_clique and checkIfClique(estimate, A) and len(estimate_indices) == K:
                        results[i][j].append(time_res["iterations"])
                        realizations_done_count += 1
                        print(
                            f"Other clique {realizations_done_count} recovered (N: {N}, K_tilde: {K_tilde})")
                    else:
                        estimate_is_clique = checkIfClique(estimate, A)
                        print("WARNING: planted clique not recovered")
                        print("Estimate is clique:",
                              estimate_is_clique)
                        print("Estimate has size K",
                              len(estimate_indices) == K)
                        if estimate_is_clique:
                            common_elements = float(
                                len([i for i in estimate_indices if i in truth])) / K
                            print("Estimate common elements with planted clique:", round(
                                common_elements * 100, 1), "%")
    filename_suffix = datetime.now().isoformat()
    filename_suffix = filename_suffix.replace(":", "-")
    if "." in filename_suffix:
        filename_suffix = filename_suffix[:filename_suffix.index(".")]
        filename_suffix = filename_suffix.replace(".", "-")
    with open(f"final_results_time_of_convergence_changing_N_multipleK_from_{Ns[0]}_to_{Ns[-1]}_{n_samples}samples_{filename_suffix}.npy", "wb") as f:
        for i in range(len(Ns)):
            np.save(f, np.array(results[i]))
        np.save(f, np.array(Ns))
        np.save(f, np.array(K_tildes))
    print(
        "Saved:", f"final_results_time_of_convergence_changing_N_multipleK_from_{Ns[0]}_to_{Ns[-1]}_{n_samples}samples_{filename_suffix}.npy")
    if send_result_email:
        status = send_email("yan-lhomme@outlook.com", "Results time of convergence",
                            f"final_results_time_of_convergence_changing_N_multipleK_from_{Ns[0]}_to_{Ns[-1]}_{n_samples}samples_{filename_suffix}.npy", pwd)
        print("Sent email:", status)


# Plots


PLOTS_COLORS = ["darkblue", "blue", "dodgerblue",
                "royalblue", "steelblue", "skyblue", "lightblue"]


def createPlotTimeOfConvergenceChangingK_N700_to_N1000(subplots=True, with_line=False, log_y=False):
    filename = f"results_time_of_convergence_finished_small_14_samples_with_easy"
    Ns = [700, 800, 900, 1000]
    K_tildes_N700 = [2.09, 1.89, 1.69, 1.59]
    K_tildes_N800 = [2.06, 1.86, 1.76, 1.66, 1.56]
    K_tildes_N900 = [2.13, 1.93, 1.83, 1.73, 1.63]
    K_tildes_N1000 = [2.21, 2.01, 1.91, 1.81, 1.71, 1.6]
    K_tildes = [K_tildes_N700, K_tildes_N800, K_tildes_N900, K_tildes_N1000]
    colors = [x for i, x in enumerate(PLOTS_COLORS) if i % 2 == 0]
    if subplots:
        figure, axis = plt.subplots(2, ceil(len(Ns) * 0.5))
        label = f"Size of the planted clique K tilde"
        with open(f"{filename}.npy", "rb") as f:
            for i, N in enumerate(Ns):
                iterations = np.load(f)
                iterations_mean = iterations.mean(axis=1)
                iterations_std_dev = iterations.std(axis=1)
                row = 0
                col = i
                if i >= ceil(len(Ns) * 0.5):
                    row = 1
                    col = i - ceil(len(Ns) * 0.5)
                if with_line:
                    axis[row, col].errorbar(np.array(K_tildes[i]), np.array(iterations_mean),
                                            yerr=np.array(iterations_std_dev), fmt=f"x", color=colors[i], ecolor=colors[i], capsize=3, ls="-")
                else:
                    axis[row, col].errorbar(np.array(K_tildes[i]), np.array(iterations_mean),
                                            yerr=np.array(iterations_std_dev), fmt=f"x", color=colors[i], ecolor=colors[i], capsize=3)
                axis[row, col].set(
                    xlabel=label, ylabel=f"Number of PT steps needed for N={N}")
    else:
        lines = []
        legends = []
        label = f"Size of the planted clique K tilde"
        with open(f"{filename}.npy", "rb") as f:
            for i, N in enumerate(Ns):
                iterations = np.load(f)
                iterations_mean = iterations.mean(axis=1)
                iterations_std_dev = iterations.std(axis=1)
                if with_line:
                    line, caplines, barlinecols = plt.errorbar(np.array(K_tildes[i]), np.array(iterations_mean),
                                                               yerr=np.array(iterations_std_dev), fmt=f"x", color=colors[i], ecolor=colors[i], capsize=3, ls="-")
                else:
                    line, caplines, barlinecols = plt.errorbar(np.array(K_tildes[i]), np.array(iterations_mean),
                                                               yerr=np.array(iterations_std_dev), fmt=f"x", color=colors[i], ecolor=colors[i], capsize=3)
                lines.append(line)
                legends.append(f"N={N}")
        label = f"Size of the planted clique K tilde (N={Ns[0]} to {Ns[-1]})"
        plt.xlabel(label)
        plt.ylabel("Number of PT steps needed")
        plt.legend(lines, legends)
    if log_y:
        plt.yscale("log")
    plt.show()


def createPlotTimeOfConvergence_N100_to_N4000_K_BP(interpolate_line=False, log_y=False):
    n_samples_big = 5
    n_samples_small = 14
    filename_big = f"PT_steps_N2000_N3000_N4000_5samples_index_0"
    filename_small = f"results_time_of_convergence_finished_small_14_samples_with_easy"
    filename_reallysmall = "results_time_of_convergence_finished__reallysmall_20_samples"
    Ns_big = [2000, 3000, 4000]
    Ns_small = [700, 800, 900, 1000]
    Ns_reallysmall = [100, 200, 300, 400, 500]
    values_means_1 = []
    values_std_1 = []
    with open(f"{filename_reallysmall}.npy", "rb") as f:
        for i, N in enumerate(Ns_reallysmall):
            iterations = np.load(f)
            iterations_mean = iterations.mean(axis=1)
            iterations_std_dev = iterations.std(axis=1)
            values_means_1.append(iterations_mean[0])
            values_std_1.append(iterations_std_dev[0])
    with open(f"{filename_small}.npy", "rb") as f:
        for i, N in enumerate(Ns_small):
            iterations = np.load(f)
            iterations_mean = iterations.mean(axis=1)
            iterations_std_dev = iterations.std(axis=1)
            values_means_1.append(iterations_mean[2])
            values_std_1.append(iterations_std_dev[2])
    with open(f"{filename_big}.npy", "rb") as f:
        for i, N in enumerate(Ns_big):
            iterations = np.load(f)
            iterations_mean = iterations.mean(axis=1)
            iterations_std_dev = iterations.std(axis=1)
            values_means_1.append(iterations_mean[0])
            values_std_1.append(iterations_std_dev[0])

    plotline1, caplines1, barlinecols1 = plt.errorbar(np.array([*Ns_reallysmall, *Ns_small, *Ns_big]), np.array(values_means_1),
                                                      yerr=np.array(values_std_1), fmt=f"xb", ecolor="b", capsize=3, lolims=False)
    caplines1[0].set_marker("_")

    if interpolate_line:
        poly_fit = Polynomial.fit(
            np.array([*Ns_reallysmall, *Ns_small, *Ns_big]), np.array(values_means_1), 4, [0, 20000])
        poly_fit_x, poly_fit_y = poly_fit.linspace(
            10000, [Ns_reallysmall[0], Ns_big[-1]])
        plt.plot(poly_fit_x, poly_fit_y)

    if log_y:
        plt.yscale("log")
    label = f"N (each point is over {n_samples_small} ({n_samples_big} for N >= 2000) samples)"
    plt.xlabel(label)
    plt.ylabel("Number of PT steps needed")
    plt.legend([plotline1], [
               "Start of hard phase (K_BP)"])
    plt.show()


def createPlotFoundNodesEvolutionPerBeta_N100_K24():
    filename = "correct_evolution_over_steps.npy"
    figure, axis = plt.subplots(4, 2)
    with open(filename, "rb") as f:
        corrects = np.load(f)
        for i in range(7):
            row = floor(i / 2)
            col = i % 2
            axis[row, col].plot(np.array(
                [j + 1 for j in range(corrects.shape[1])]), corrects[i, :], ".", color=PLOTS_COLORS[i])
            axis[row, col].set(
                xlabel="PT steps", ylabel="Correct nodes")
            axis[row, col].set_title(
                f"Number of correct nodes in estimate for beta={round(1.0 - 0.15 * i, 2)}")
    axis[3, 1].remove()
    plt.show()


def createPlotTimeOfConvergence_N700_to_N3000_K_BP(log_y=False):
    index_color = 2
    n_samples = 8
    Ns_K_BPs = [700, 800, 900, 1000, 2000, 3000]
    N_700_K_16 = [1817, 6621, 2221, 13483, 8413, 5235, 36295, 3561]
    N_800_K_17 = [6695, 541, 299, 5087, 15555, 1003, 5441, 975]
    N_900_K_18 = [7623, 5843, 1789, 3083, 3103, 1273, 5683, 1975]
    N_1000_K_19 = [2635, 13111, 7931, 7031, 13515, 26049, 7379, 33679]
    N_2000_K_27 = [10699, 78087, 7345, 36023, 28527, 8743, 134253, 39411]
    N_3000_K_33 = [79673, 75225, 63127]
    results = np.array(
        [N_700_K_16, N_800_K_17, N_900_K_18, N_1000_K_19, N_2000_K_27])
    results_mean = np.array(
        [*list(results.mean(axis=1)), np.array(N_3000_K_33).mean()])
    results_std_dev = np.array(
        [*list(results.std(axis=1)), np.array(N_3000_K_33).std()])

    A = np.vstack([np.array(Ns_K_BPs), np.ones(len(Ns_K_BPs))]).T
    m, c = np.linalg.lstsq(A, results_mean, rcond=None)[0]

    plotline1, caplines1, barlinecols1 = plt.errorbar(np.array(Ns_K_BPs), np.array(results_mean),
                                                      yerr=np.array(results_std_dev), fmt=f"x", color=PLOTS_COLORS[index_color], ecolor=PLOTS_COLORS[index_color], capsize=3, lolims=False)
    caplines1[0].set_marker("_")
    interp_x = np.array([Ns_K_BPs[0] - 100, *Ns_K_BPs, Ns_K_BPs[-1] + 100])
    plt.plot(interp_x, m * np.array(interp_x) +
             c, color=PLOTS_COLORS[-1])
    if log_y:
        plt.yscale("log")
    label = f"N (each point is over {n_samples} samples)"
    plt.xlabel(label)
    plt.ylabel("Number of PT steps needed")
    plt.legend([plotline1], [
               "Start of hard phase (K_BP = floor(sqrt(N/e)))"])
    plt.show()


def createPlotTimeOfConvergence_N700_to_N2000_K_BPminus1(log_y=False):
    index_color = 2
    n_samples = 8
    Ns_K_BPs = [700, 800, 900, 1000, 2000]
    N_700_K_15 = [11809, 59827, 55941, 11920, 10235, 15500, 10445, 2237]
    N_800_K_16 = [11395, 653, 3677, 25073, 23239, 24289, 4380, 490]
    N_900_K_17 = [6659, 3261, 18815, 8847, 9049, 1871, 14863, 9993]
    N_1000_K_18 = [7397, 12455, 1333, 2801, 3749, 44313, 1805, 1793]
    N_2000_K_26 = [87447, 17005, 45527, 12555, 51627, 293851, 112189]
    results = np.array(
        [N_700_K_15, N_800_K_16, N_900_K_17, N_1000_K_18])
    results_mean = np.array(
        [*list(results.mean(axis=1)), np.array(N_2000_K_26).mean()])
    results_std_dev = np.array(
        [*list(results.std(axis=1)), np.array(N_2000_K_26).std()])

    A = np.vstack([np.array(Ns_K_BPs), np.ones(len(Ns_K_BPs))]).T
    m, c = np.linalg.lstsq(A, results_mean, rcond=None)[0]

    plotline1, caplines1, barlinecols1 = plt.errorbar(np.array(Ns_K_BPs), np.array(results_mean),
                                                      yerr=np.array(results_std_dev), fmt=f"x", color=PLOTS_COLORS[index_color], ecolor=PLOTS_COLORS[index_color], capsize=3, lolims=False)
    caplines1[0].set_marker("_")
    interp_x = np.array([Ns_K_BPs[0] - 100, *Ns_K_BPs, Ns_K_BPs[-1] + 100])
    plt.plot(interp_x, m * np.array(interp_x) +
             c, color=PLOTS_COLORS[-1])
    if log_y:
        plt.yscale("log")
    label = f"N (each point is over {n_samples} samples)"
    plt.xlabel(label)
    plt.ylabel("Number of PT steps needed")
    plt.legend([plotline1], [
               "Hard phase (K_BP - 1 = floor(sqrt(N/e))) - 1"])
    plt.show()


def createPlotTimeOfConvergence_N700_to_N4000_KsaroundHard(log_y=False):
    # order in the file: K_BP, K_BP - 1, K_BP - 2 (no 700), K_BP + 1, K_BP + 2
    index_color = 2
    index_color_minus1 = 0
    n_samples = 8
    # Ns_K_BPs = [700, 800, 900, 1000, 2000, 3000, 4000]
    Ns_K_BPs = [700, 800, 900, 1000, 2000, 3000]
    N_700 = [5780, 31207, 9300, 7340, 11401, 1532, 1633, 7760]
    N_800 = [48175, 10850, 32620, 169, 18369, 9017, 14512, 4578]
    N_900 = [9382, 7778, 8013, 7004, 3483, 3217, 10978, 5537]
    N_1000 = [13054, 3719, 2313, 34678, 5247, 37531, 27964, 1423]
    N_2000 = [77429, 32938, 25089, 258160, 125528, 103862, 10709, 20733]
    N_3000 = [35451, 117622, 25345, 88763, 30099, 369286, 38701, 196546]
    results = np.array([N_700, N_800, N_900, N_1000, N_2000, N_3000])
    results_mean = results.mean(axis=1)
    results_std_dev = results.std(axis=1)

    N_700_minus1 = [3194, 12774, 22961, 18578, 68346, 31217, 43, 203644]
    N_1000_minus1 = [59467, 47912, 29708, 34077, 8362, 4069, 3304, 131330]
    N_1000_minus2 = [5603, 55456, 12262, 141487, 44334, 87524, 5363, 58663]
    N_2000_minus1 = [137638, 2501, 65045, 12750, 128608, 48165, 119159]
    N_2000_minus1.append(np.array(N_2000_minus1).mean())
    results_minus1 = np.array([N_700_minus1, N_1000_minus1, N_2000_minus1])
    results_minus1_mean = results_minus1.mean(axis=1)
    results_minus1_std_dev = results_minus1.std(axis=1)

    plotline1, caplines1, barlinecols1 = plt.errorbar(np.array(Ns_K_BPs), np.array(results_mean),
                                                      yerr=np.array(results_std_dev), fmt=f"x", color=PLOTS_COLORS[index_color], ecolor=PLOTS_COLORS[index_color], capsize=3, lolims=False)
    caplines1[0].set_marker("_")

    plotline2, caplines2, barlinecols2 = plt.errorbar(np.array([700, 1000, 2000]), np.array(results_minus1_mean),
                                                      yerr=np.array(results_minus1_std_dev), fmt=f"x", color=PLOTS_COLORS[index_color_minus1], ecolor=PLOTS_COLORS[index_color_minus1], capsize=3, lolims=False)
    caplines2[0].set_marker("_")

    if log_y:
        plt.yscale("log")
    label = f"N (each point is over {n_samples} samples)"
    plt.xlabel(label)
    plt.ylabel("Number of PT steps needed")
    plt.legend([plotline1, plotline2], [
               "Start of hard phase (K_BP = floor(sqrt(N/e)))", "Hard phase (K_BP - 1)"])
    plt.show()


def createPlotTimeOfConvergenceChangingK():
    n_samples = 8

    N_700_K_15 = [11809, 59827, 55941, 11920, 10235, 15500, 10445, 2237]
    N_700_K_16 = [1817, 6621, 2221, 13483, 8413, 5235, 36295, 3561]
    N_700_K_17 = [6925, 2797, 4968, 15381, 19810, 4542, 6408, 15007]
    N_700_K_18 = [5067, 473, 3716, 9975, 7245, 10883, 111, 289]

    N_800_K_15 = [13461, 11293, 5471, 8675, 16985, 5593, 20468, 8632]
    N_800_K_16 = [11395, 653, 3677, 25073, 23239, 24289, 4380, 490]
    N_800_K_17 = [6695, 541, 299, 5087, 15555, 1003, 5441, 975]

    N_900_K_16 = [47600, 26943, 96071, 57141, 120170, 26031, 1155, 23086]
    N_900_K_17 = [6659, 3261, 18815, 8847, 9049, 1871, 14863, 9993]
    N_900_K_18 = [7623, 5843, 1789, 3083, 3103, 1273, 5683, 1975]

    N_1000_K_17 = [42649, 36621, 17643, 5991, 171615, 64213, 35629, 1635]
    N_1000_K_18 = [7397, 12455, 1333, 2801, 3749, 44313, 1805, 1793]
    N_1000_K_19 = [2635, 13111, 7931, 7031, 13515, 26049, 7379, 33679]
    N_1000_K_20 = [2837, 5133, 6717, 3385, 23493, 9401, 20775, 18742]
    N_1000_K_21 = [7633, 1875, 1075, 67, 617, 2237, 4455, 445]

    K_tildes_N700 = [round(K / np.log2(700), 2) for K in [15, 16, 17, 18]]
    K_tildes_N800 = [round(K / np.log2(800), 2) for K in [15, 16, 17]]
    K_tildes_N900 = [round(K / np.log2(900), 2) for K in [16, 17, 18]]
    K_tildes_N1000 = [round(K / np.log2(1000), 2)
                      for K in [17, 18, 19, 20, 21]]

    N_700 = np.array([N_700_K_15, N_700_K_16, N_700_K_17, N_700_K_18])
    N_800 = np.array([N_800_K_15, N_800_K_16, N_800_K_17])
    N_900 = np.array([N_900_K_16, N_900_K_17, N_900_K_18])
    N_1000 = np.array(
        [N_1000_K_17, N_1000_K_18, N_1000_K_19, N_1000_K_20, N_1000_K_21])

    means_700 = N_700.mean(axis=1)
    means_800 = N_800.mean(axis=1)
    means_900 = N_900.mean(axis=1)
    means_1000 = N_1000.mean(axis=1)

    std_700 = N_700.std(axis=1)
    std_800 = N_800.std(axis=1)
    std_900 = N_900.std(axis=1)
    std_1000 = N_1000.std(axis=1)

    plotline1, caplines1, barlinecols1 = plt.errorbar(np.array(K_tildes_N700), means_700,
                                                      yerr=std_700, fmt=f"x", color=PLOTS_COLORS[0], ecolor=PLOTS_COLORS[0], capsize=3, lolims=False, ls="-")
    caplines1[0].set_marker("_")
    plotline2, caplines2, barlinecols2 = plt.errorbar(np.array(K_tildes_N800), means_800,
                                                      yerr=std_800, fmt=f"x", color=PLOTS_COLORS[2], ecolor=PLOTS_COLORS[2], capsize=3, lolims=False, ls="-")
    caplines2[0].set_marker("_")
    plotline3, caplines3, barlinecols3 = plt.errorbar(np.array(K_tildes_N900), means_900,
                                                      yerr=std_900, fmt=f"x", color=PLOTS_COLORS[4], ecolor=PLOTS_COLORS[4], capsize=3, lolims=False, ls="-")
    caplines3[0].set_marker("_")
    plotline4, caplines4, barlinecols4 = plt.errorbar(np.array(K_tildes_N1000), means_1000,
                                                      yerr=std_1000, fmt=f"x", color=PLOTS_COLORS[6], ecolor=PLOTS_COLORS[6], capsize=3, lolims=False, ls="-")
    caplines4[0].set_marker("_")
    # plt.yscale("log")
    label = f"K_tilde (each point is over {n_samples} samples)"
    plt.xlabel(label)
    plt.ylabel("Number of PT steps needed")
    plt.legend([plotline1, plotline2, plotline3, plotline4], [
               "N = 700", "N = 800", "N = 900", "N = 1000"])
    plt.show()


def parseLogFile(log_filename, file_out, file_out_list):
    count = 0
    current = ""
    if len(file_out_list) > 0:
        with open(file_out_list, "w") as f_out:
            with open(log_filename, "r") as f:
                lines = f.readlines()
                for l in lines:
                    if len(current) > 0:
                        if "WARNING" not in l:
                            count += 1
                            if count < 8:
                                f_out.write(f"{current}, ")
                            else:
                                f_out.write(f"{current}")
                        current = ""
                    elif "N " in l:
                        count = 0
                        val = l.replace(" ", "_").replace("\n", "_")
                        f_out.write(f"]\n{val}")
                    elif "K " in l and "True" not in l:
                        val = l.replace(" ", "_").replace("\n", " = ")
                        f_out.write(f"{val}[")
                    elif "#" in l:
                        steps = l[l.index("#") + 1:]
                        steps = steps[:steps.index(",")]
                        current = steps
    count = 0
    current = ""
    if len(file_out) > 0:
        with open(file_out, "w") as f_out:
            with open(log_filename, "r") as f:
                lines = f.readlines()
                for l in lines:
                    if len(current) > 0:
                        if "WARNING" not in l:
                            count += 1
                            if count < 8:
                                f_out.write(f"{current}, ")
                            else:
                                f_out.write(f"{current}")
                        current = ""
                    elif "N " in l:
                        count = 0
                        f_out.write(f"]\n{l}")
                    elif "K " in l and "True" not in l:
                        f_out.write(f"{l}[")
                    elif "#" in l:
                        steps = l[l.index("#") + 1:]
                        steps = steps[:steps.index(",")]
                        current = steps
    if len(file_out) > 0:
        if len(file_out_list) > 0:
            print("Files", file_out, file_out_list, "created")
        else:
            print("File", file_out, "created")
    elif len(file_out_list) > 0:
        print("File", file_out_list, "created")


if __name__ == '__main__':
    # ===============================
    # To test the clique recovery with PT uncomment this section and change the values of K_tilde_test (or K_test), N_test according to your needs
    # K_tilde_test = 1.92
    # N_test = 1000
    # K_test = getKFromKTilde(N_test, K_tilde_test)
    # testParallelTempering(N_test, K_test, show_graph=False)
    # ===============================

    # ===============================
    # To sample the convergence of PT by changing N uncomment this section
    # Ns = [200, 500, 1000, 2000, 3000, 4000, 5000]  # the N's to be sampled
    # # size of the planted clique K with one of the following options
    # K_as_list = [8, 13, 19, 27, 33, 38, 42] # each N has its own K ([] if not this option)
    # K_tilde = 1.92 # K = K_tilde * log_2(N) (-1 if not this option)
    # K_to_N_factor = 0.125 # K = N * K_to_N_factor
    # n_samples = 5  # number of graph realizations per N
    # timeOfConvergenceChangingN(Ns, n_samples, K_as_list=K_as_list, K_tilde=K_tilde, K_to_N_factor=K_to_N_factor)
    # ===============================

    # ===============================
    # To sample the convergence of PT by changing N and multiple K's by N uncomment this section
    # Ns = [200, 500, 1000, 2000, 3000, 4000, 5000]  # the N's to be sampled
    # # sizes of the planted clique K to be samples by N
    # K_tildes = [[floor(np.sqrt(N/np.e)) - i for i in range(2)] for N in Ns]
    # n_samples = 5  # number of graph realizations per N
    # timeOfConvergenceChangingK(Ns, K_tildes, n_samples=n_samples)
    # ===============================

    # ===============================
    # Create plots from data
    # createPlotTimeOfConvergenceChangingK_N700_to_N1000(subplots=True, with_line=False, log_y=False)
    # createPlotTimeOfConvergence_N100_to_N4000_K_BP(interpolate_line=False, log_y=False)
    # createPlotFoundNodesEvolutionPerBeta_N100_K24()
    # createPlotTimeOfConvergence_N700_to_N3000_K_BP(log_y=False)
    # createPlotTimeOfConvergence_N700_to_N2000_K_BPminus1(log_y=False)
    # createPlotTimeOfConvergence_N700_to_N4000_KsaroundHard(log_y=False)
    # createPlotTimeOfConvergenceChangingK()
    # ===============================

    # ===============================
    # TESTS

    # Paper short
    # Ns = [2000, 3000, 4000, 5000]
    # K_tildes_N2000 = [1.64, 1.82, 2.01,
    #               2.18, 2.37, 2.55, 2.73]
    # K_tildes_N3000 = [1.73, 1.9, 2.07, 2.25, 2.43, 2.6, 2.77, 2.86]
    # K_tildes_N4000 = [1.92, 2.09, 2.26, 2.42,
    #                 2.59, 2.76, 2.92, 3.09, 3.26]
    # K_tildes_N5000 = [2.19, 2.36, 2.52, 2.68,
    #                 2.85, 3.01, 3.17, 3.34, 3.5]
    # K_tildes = [K_tildes_N2000, K_tildes_N3000, K_tildes_N4000, K_tildes_N5000]
    # n_samples = 5  # number of graph realizations per N
    # timeOfConvergenceChangingK(Ns, K_tildes, n_samples=n_samples)

    # Small Ns
    # Ns = [700, 800, 900, 1000]
    # K_tildes_N700 = [2.09, 1.89, 1.69, 1.59]
    # K_tildes_N800 = [2.06, 1.86, 1.76, 1.66, 1.56]
    # K_tildes_N900 = [2.13, 1.93, 1.83, 1.73, 1.63]
    # K_tildes_N1000 = [2.21, 2.01, 1.91, 1.81, 1.71, 1.6]
    # K_tildes = [K_tildes_N700, K_tildes_N800, K_tildes_N900, K_tildes_N1000]
    # n_samples = 14  # number of graph realizations per N
    # timeOfConvergenceChangingK(Ns, K_tildes, n_samples=n_samples)

    # Hard phase
    # Ns = [700, 1000, 2000, 3000, 4000]
    # K_tildes = [[float(floor(np.sqrt(N/np.e)) - i) / np.log2(N)
    #              for i in range(3 if N != 700 else 2)] for N in Ns]
    # n_samples = 8  # number of graph realizations per N
    # timeOfConvergenceChangingK(Ns, K_tildes, n_samples=n_samples)

    # Easy phase
    # Ns = [700, 1000, 2000, 3000, 4000]
    # K_tildes = [[float(floor(np.sqrt(N/np.e)) + i) / np.log2(N)
    #              for i in range(1, 3)] for N in Ns]
    # n_samples = 8  # number of graph realizations per N
    # timeOfConvergenceChangingK(Ns, K_tildes, n_samples=n_samples)

    # Hard phase complete
    # Ns = [700, 800, 900, 1000, 2000, 3000, 4000]
    # K_tildes = []
    # K_tildes.append([floor(np.sqrt(700/np.e)), floor(np.sqrt(700/np.e)) - 1,
    #                 floor(np.sqrt(700/np.e)) + 1, floor(np.sqrt(700/np.e)) + 2])
    # K_tildes.append([floor(np.sqrt(800/np.e))])
    # K_tildes.append([floor(np.sqrt(900/np.e))])
    # K_tildes.append([floor(np.sqrt(1000/np.e)), floor(np.sqrt(1000/np.e)) - 1, floor(
    #     np.sqrt(1000/np.e)) - 2, floor(np.sqrt(1000/np.e)) + 1, floor(np.sqrt(1000/np.e)) + 2])
    # K_tildes.append([floor(np.sqrt(2000/np.e)), floor(np.sqrt(2000/np.e)) - 1, floor(
    #     np.sqrt(2000/np.e)) - 2, floor(np.sqrt(2000/np.e)) + 1, floor(np.sqrt(2000/np.e)) + 2])
    # K_tildes.append([floor(np.sqrt(3000/np.e)), floor(np.sqrt(3000/np.e)) - 1, floor(
    #     np.sqrt(3000/np.e)) - 2, floor(np.sqrt(3000/np.e)) + 1, floor(np.sqrt(3000/np.e)) + 2])
    # K_tildes.append([floor(np.sqrt(4000/np.e)), floor(np.sqrt(4000/np.e)) - 1, floor(
    #     np.sqrt(4000/np.e)) - 2, floor(np.sqrt(4000/np.e)) + 1, floor(np.sqrt(4000/np.e)) + 2])
    # for i in range(len(K_tildes)):
    #     for j in range(len(K_tildes[i])):
    #         K_tildes[i][j] = float(K_tildes[i][j]) / np.log2(Ns[i])
    # n_samples = 8  # number of graph realizations per N
    # Ns.reverse()
    # K_tildes.reverse()
    # print("Sampling threshold 0.66")
    # timeOfConvergenceChangingK(Ns, K_tildes, n_samples=n_samples)

    # Hard phase complete 5000
    # Ns = [5000]
    # K_tildes = []
    # K_tildes.append([floor(np.sqrt(5000/np.e)), floor(np.sqrt(5000/np.e)) - 1, floor(
    #     np.sqrt(5000/np.e)) - 2, floor(np.sqrt(5000/np.e)) + 1, floor(np.sqrt(5000/np.e)) + 2, *[37, 35, 33, 31, 29, 27]])
    # for i in range(len(K_tildes)):
    #     for j in range(len(K_tildes[i])):
    #         K_tildes[i][j] = float(K_tildes[i][j]) / np.log2(Ns[i])
    # n_samples = 8  # number of graph realizations per N
    # print("Sampling all hard phase for N=5000 (method: > threshold and % 2)")
    # timeOfConvergenceChangingK(Ns, K_tildes, n_samples=n_samples)

    # Hard phase paper's results completion
    # Ns = [2000, 3000, 4000]
    # K_tildes = [[22, 20, 18], [28, 26, 24, 22, 20], [33, 31, 29, 27, 25, 23]]
    # for i in range(len(K_tildes)):
    #     for j in range(len(K_tildes[i])):
    #         K_tildes[i][j] = float(K_tildes[i][j]) / np.log2(Ns[i])
    # n_samples = 8  # number of graph realizations per N
    # timeOfConvergenceChangingK(
    #     Ns, K_tildes, n_samples=n_samples)

    # Hard phase complete
    # Ns = [3000]
    # K_tildes = []
    # K_tildes.append([floor(np.sqrt(3000/np.e)), floor(np.sqrt(3000/np.e)) - 1, floor(
    #     np.sqrt(3000/np.e)) - 2, floor(np.sqrt(3000/np.e)) + 1, floor(np.sqrt(3000/np.e)) + 2])
    # for i in range(len(K_tildes)):
    #     for j in range(len(K_tildes[i])):
    #         K_tildes[i][j] = float(K_tildes[i][j]) / np.log2(Ns[i])
    # n_samples = 8  # number of graph realizations per N
    # print("Sampling only 3000 threshold 0.66")
    # timeOfConvergenceChangingK(Ns, K_tildes, n_samples=n_samples)

    # Ns = [800, 900, 1000]
    # K_tildes = []
    # K_tildes.append([floor(np.sqrt(800/np.e)) - 1,
    #                 floor(np.sqrt(800/np.e)) - 2])
    # K_tildes.append([floor(np.sqrt(900/np.e)) - 1,
    #                 floor(np.sqrt(900/np.e)) - 2])
    # K_tildes.append([floor(np.sqrt(1000/np.e)) - 3])
    # for i in range(len(K_tildes)):
    #     for j in range(len(K_tildes[i])):
    #         K_tildes[i][j] = float(K_tildes[i][j]) / np.log2(Ns[i])
    # n_samples = 8  # number of graph realizations per N
    # print("Sampling only rest threshold 0.66")
    # timeOfConvergenceChangingK(Ns, K_tildes, n_samples=n_samples)

    # ada-14
    Ns = [700, 800, 900, 1000, 2000, 3000, 4000]
    K_tildes = [[floor(np.sqrt(N/np.e)) / np.log2(N)] for N in Ns]
    n_samples = 4
    print("700 to 4000, 4 samples, at K_BP, threshold: 0.58")
    timeOfConvergenceChangingK(
        Ns, K_tildes, n_samples=n_samples, send_result_email=False)

    # ===============================

    # x = np.linspace(Ns[0], Ns[-1], Ns[-1] - Ns[0])
    # y_sqrt = np.sqrt(x / np.e)
    # y_log16 = 1.6 * np.log2(x)
    # y_log2 = 2 * np.log2(x)
    # plt.plot(x, y_sqrt, color="b")
    # plt.plot(x, y_log2, color="r")
    # plt.plot(x, y_log16, color="orange")
    # plt.show()

    # print([floor(np.sqrt(N/np.e)) for N in [700, 800, 900, 1000, 2000, 3000]])

    # ssh access: ssh pytharski -l lhommey
    pass


"""
NOTES:
for hard phase index 0 I used betas with 0.15 intervals from N=5000


really small Ns
Ns = [100, 200, 300, 400, 500]
Ks = [13, 15, 16, 17, 17]


K_BP changing N
Ns_K_BPs = [700, 800, 900, 1000, 2000, 3000, 4000]
K_BPs = [floor(np.sqrt(N/np.e)) for N in Ns_K_BPs]
results = [[] for _ in range(len(Ns_K_BPs))]

paper:
K_tildes_N2000 = [1.64, 1.82, 2.01,
                  2.18, 2.37, 2.55, 2.73]
K_tildes_N3000 = [1.73, 1.9, 2.07, 2.25, 2.43, 2.6, 2.77, 2.86]
K_tildes_N4000 = [1.92, 2.09, 2.26, 2.42,
                  2.59, 2.76, 2.92, 3.09, 3.26]
K_tildes_N5000 = [2.19, 2.36, 2.52, 2.68,
                  2.85, 3.01, 3.17, 3.34, 3.5]
# K_tildes_N2000 = [1.64, 1.73, 1.82, 1.92, 2.01,
#                   2.09, 2.18, 2.28, 2.37, 2.46, 2.55, 2.64, 2.73]
# K_tildes_N3000 = [1.73, 1.82, 1.9, 1.99, 2.07, 2.16,
#                   2.25, 2.34, 2.43, 2.51, 2.6, 2.68, 2.77, 2.86]
# K_tildes_N4000 = [1.92, 2.01, 2.09, 2.17, 2.26, 2.34, 2.42,
#                   2.51, 2.59, 2.67, 2.76, 2.84, 2.92, 3.01, 3.09, 3.17, 3.26]
# K_tildes_N5000 = [2.19, 2.28, 2.36, 2.44, 2.52, 2.61, 2.68,
#                   2.77, 2.85, 2.93, 3.01, 3.09, 3.17, 3.26, 3.34, 3.42, 3.5]
"""


"""
Old functions

def createPlotTimeOfConvergence_N700_to_N4000_K_BP(log_y=False):
    index_color = 2
    n_samples = 8
    # Ns_K_BPs = [700, 800, 900, 1000, 2000, 3000, 4000]
    Ns_K_BPs = [700, 800, 900, 1000, 2000, 3000]
    N_700 = [5780, 31207, 9300, 7340, 11401, 1532, 1633, 7760]
    N_800 = [48175, 10850, 32620, 169, 18369, 9017, 14512, 4578]
    N_900 = [9382, 7778, 8013, 7004, 3483, 3217, 10978, 5537]
    N_1000 = [13054, 3719, 2313, 34678, 5247, 37531, 27964, 1423]
    N_2000 = [77429, 32938, 25089, 258160, 125528, 103862, 10709, 20733]
    N_3000 = [35451, 117622, 25345, 88763, 30099, 369286, 38701, 196546]
    results = np.array([N_700, N_800, N_900, N_1000, N_2000, N_3000])
    results_mean = results.mean(axis=1)
    results_std_dev = results.std(axis=1)
    # filename = f"results_K_BP"
    # for N in Ns_K_BPs:
    #     filename += f"_{N}"
    # with open(f"{filename}.npy", "rb") as f:
    #     results = np.load(f)
    #     results_mean = results.mean(axis=1)
    #     results_std_dev = results.std(axis=1)

    plotline1, caplines1, barlinecols1 = plt.errorbar(np.array(Ns_K_BPs), np.array(results_mean),
                                                      yerr=np.array(results_std_dev), fmt=f"x", color=PLOTS_COLORS[index_color], ecolor=PLOTS_COLORS[index_color], capsize=3, lolims=False)
    caplines1[0].set_marker("_")
    if log_y:
        plt.yscale("log")
    label = f"N (each point is over {n_samples} samples)"
    plt.xlabel(label)
    plt.ylabel("Number of PT steps needed")
    plt.legend([plotline1], [
               "Start of hard phase (K_BP = floor(sqrt(N/e)))"])
    plt.show()

def drawCandidate(x, N, method="switch_1", p=0.5, k=1):
    # methods:
    #    switch_1: switch 1 element of x with probability p
    #    switch_k: switch k element of x independently with probability p
    # Old method that did not take into account the fact that the inserted elements have to be linked to the other elements of the current clique
    
    x_candidate = np.copy(x)
    if method in ["switch_1", "switch_k"]:
        if method == "switch_1":
            k = 1
        choice = np.random.choice(N, k, replace=True)
        for i in choice:
            p_switch = np.random.uniform()
            if p_switch < p:
                x_candidate[i] = 1 - x_candidate[i]
        return x_candidate
    return x_candidate


def drawCandidateWithCliqueCheck(x, N, A, method="switch_1", p=0.5, k=1):
    # methods:
    #    switch_1: switch 1 element of x with probability p
    #    switch_k: switch k element of x independently with probability p
    # Ensure that all inserted elements are linked to the other elements of the current clique
    # return a candidate (np array with entries 0 or 1)
    # (Computation really slow use one of the drawCandidateWithCliqueCheckAndAddingOnlyPossibleNodes methods instead)
    
    x_candidate = np.copy(x)
    if method in ["switch_1", "switch_k"]:
        if method == "switch_1":
            k = 1
        if N < k:
            k = N
        choice = np.random.choice(N, k, replace=False)
        for i in choice:
            p_switch = np.random.uniform()
            if p_switch < p:
                if x_candidate[i] == 1:
                    x_candidate[i] = 0
                else:
                    if len([j for j in range(N) if j != i and x_candidate[j] == 1 and A[i, j] != 1]) == 0:
                        x_candidate[i] = 1
        return x_candidate
    return x_candidate


def drawCandidateWithCliqueCheckAndAddingOnlyPossibleNodes_1(x, N, K, A, method="switch_1", p=0.5, k=1, param_remove=0.5):
    # methods:
    #    switch_1: switch 1 element of x with probability p
    #    switch_k: switch k element of x independently with probability p
    # Ensure that all inserted elements are linked to the other elements of the current clique
    # Candidate elements (to be added) are chosen among the common neighbors of the current elements of the clique
    # return a candidate (np array with entries 0 or 1)
    # First implementation (not the fastest)
    
    x_candidate = np.copy(x)
    if method in ["switch_1", "switch_k"]:
        if method == "switch_1":
            k = 1
            if N < k:
                k = N
            choice = np.random.choice(N, k, replace=False)
            for i in choice:
                p_switch = np.random.uniform()
                if p_switch < p:
                    if x_candidate[i] == 1:
                        x_candidate[i] = 0
                    else:
                        if len([j for j in range(N) if j != i and x_candidate[j] == 1 and A[i, j] != 1]) == 0:
                            x_candidate[i] = 1
            return x_candidate
        else:
            if N < k:
                k = N
            clique_indices1 = [i for i in range(N) if x_candidate[i] == 1]
            k_remove = max(
                1, floor(len(clique_indices1) * param_remove))
            k_add = k - k_remove
            if len(clique_indices1) == 0:
                k_add = k
            else:
                if k_remove > len(clique_indices1):
                    k_remove = len(clique_indices1)
                    k_add = k - k_remove
                choice_remove = np.random.choice(
                    clique_indices1, k_remove, replace=False)
                for i in choice_remove:
                    p_switch = np.random.uniform()
                    if p_switch < p:
                        x_candidate[i] = 0
            common_neighbors = []
            clique_indices2 = [i for i in range(N) if x_candidate[i] == 1]
            common_vector = np.dot(A, x_candidate)
            clique_indices2_len = len(clique_indices2)
            for i in range(N):
                if common_vector[i] == clique_indices2_len:
                    common_neighbors.append(i)
            if k_add > len(common_neighbors):
                k_add = len(common_neighbors)
            choice_add = np.random.choice(
                common_neighbors, k_add, replace=False)
            added = []
            for i in choice_add:
                p_switch = np.random.uniform()
                if p_switch < p:
                    if len([j for j in added if A[i, j] != 1]) == 0:
                        x_candidate[i] = 1
                        added.append(i)
            return x_candidate
    return x_candidate


def drawCandidateWithCliqueCheckAndAddingOnlyPossibleNodes_2(x, N, K, A, method="switch_1", p=0.5, k=1, param_remove=0.5):
    # methods:
    #    switch_1: switch 1 element of x with probability p
    #    switch_k: switch k element of x independently with probability p
    # Ensure that all inserted elements are linked to the other elements of the current clique
    # Candidate elements (to be added) are chosen among the common neighbors of the current elements of the clique
    # return a candidate (np array with entries 0 or 1)
    # Second implementation (not the fastest)
    
    x_candidate = np.copy(x)
    if method in ["switch_1", "switch_k"]:
        if method == "switch_1":
            k = 1
            if N < k:
                k = N
            choice = np.random.choice(N, k, replace=False)
            for i in choice:
                p_switch = np.random.uniform()
                if p_switch < p:
                    if x_candidate[i] == 1:
                        x_candidate[i] = 0
                    else:
                        if len([j for j in range(N) if j != i and x_candidate[j] == 1 and A[i, j] != 1]) == 0:
                            x_candidate[i] = 1
            return x_candidate
        else:
            if N < k:
                k = N
            clique_indices1 = [i for i in range(N) if x_candidate[i] == 1]
            k_remove = max(
                1, floor(len(clique_indices1) * param_remove))
            k_add = k - k_remove
            if len(clique_indices1) == 0:
                k_add = floor(K * 0.5)
            else:
                if k_remove > len(clique_indices1):
                    k_remove = len(clique_indices1)
                    k_add = k - k_remove
                choice_remove = np.random.choice(
                    clique_indices1, k_remove, replace=False)
                for i in choice_remove:
                    p_switch = np.random.uniform()
                    if p_switch < p:
                        x_candidate[i] = 0
            not_clique_indices2 = [i for i in range(N) if x_candidate[i] == 0]
            clique_indices2 = [i for i in range(N) if x_candidate[i] == 1]
            if k_add > len(not_clique_indices2):
                k_add = len(not_clique_indices2)
            choice_add = np.random.choice(
                not_clique_indices2, k_add, replace=False)
            n_rejected = 0
            for i in choice_add:
                p_switch = np.random.uniform()
                if p_switch < p:
                    linked = True
                    for j in clique_indices2:
                        if A[i, j] != 1:
                            linked = False
                            break
                    if linked:
                        x_candidate[i] = 1
                        clique_indices2.append(i)
                    else:
                        n_rejected += 1
            if n_rejected >= 0.5 * k_add:
                not_clique_indices2 = [
                    i for i in range(N) if x_candidate[i] == 0]
                choice_add = np.random.choice(
                    not_clique_indices2, n_rejected, replace=False)
                for i in choice_add:
                    linked = True
                    for j in clique_indices2:
                        if A[i, j] != 1:
                            linked = False
                            break
                    if linked:
                        x_candidate[i] = 1
                        clique_indices2.append(i)
            return x_candidate
    return x_candidate

"""

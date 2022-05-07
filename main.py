from mails import send_email
from progressbar import printProgressBar
import concurrent.futures
from datetime import datetime
from math import ceil, factorial, floor
import numpy as np
import matplotlib.pyplot as plt


LOG_1_HALF = np.log(0.5)


def createRandomGraphWithPlantedClique(N, K, with_neighbors=False):
    """
    Return matrix A and planted clique vector v of a realization of a planted random graph of size N and planted clique size K
    if with_neighbors is True then it returns also a list of list of neighbors indices for each node
    """
    v_index_choices = np.random.choice(N, K, replace=False)
    v = np.array([1 if i in v_index_choices else 0 for i in range(N)])
    A = np.zeros((N, N))
    A_neighbors = [[] for _ in range(N)]
    for i in range(N-1):
        for j in range(i + 1, N):
            if v[i] * v[j] == 1:
                A[i, j] = 1
                A_neighbors[i].append(j)
                A_neighbors[j].append(i)
            else:
                p = np.random.uniform()
                if p < 0.5:
                    A[i, j] = 1
                    A_neighbors[i].append(j)
                    A_neighbors[j].append(i)
    A = A + A.T
    if with_neighbors:
        return A, v, A_neighbors
    return A, v


def preventAbsenceOfLinksBetweenTwoElmentsOfTheClique(N_count_constant, x):
    """
    Count the number of times log(1/2) comes out in the second sum of the energy
    It is called in this way because at the beginning it prevented to have no link between elements of the current clique but now it is handled in the drawCandidateWithCliqueCheck method 
    N_count_constant = N * (N - 1)
    """
    x_sum = x.sum()
    x_sum = x_sum * (x_sum - 1)  # number of pairs (i,j) with x[i]*x[j] == 1
    count = 0.5 * (N_count_constant - x_sum)
    return count


def H(x, N, log_K_over_N, log_1_minus_K_over_N, N_count_constant):
    """
    Compute the energy of the estimate x
    """
    count_log_1_half = preventAbsenceOfLinksBetweenTwoElmentsOfTheClique(
        N_count_constant, x)
    if count_log_1_half == -1:
        return float("inf")
    second_sum = count_log_1_half * LOG_1_HALF
    number_ones = x.sum()
    first_sum = number_ones * log_K_over_N
    first_sum += (N - number_ones) * log_1_minus_K_over_N
    return first_sum - second_sum


def drawCandidate(x, N, K, A, method="switch_standard", p=0.5, k=1, param_remove=0.5, beta=-1.0, A_neighbors=None):
    """
    methods:
        switch_standard: "standard" procedure
        switch_k: switch k element of x: k_remove = (param_remove * min(k, size of current clique)) elements are removed with probability p, (k - k_remove) from the common neighbors of the current clique x are added with probability (p + 0.25 * beta)
    Ensure that all inserted elements are linked to the other elements of the current clique
    Candidate elements (to be added) are chosen among the common neighbors of the current elements of the clique
    return a candidate (np array with entries 0 or 1), and an integer representing the order of the number of operations needed to compute the common neighbors of the current clique (0 if method is switch_standard)
    """
    x_candidate = np.copy(x)
    time_complexity = 0
    if method in ["switch_standard", "switch_k"]:
        if method == "switch_standard":
            k = 1
            if N < k:
                k = N
            param_rem = max(1, floor(0.0005 * N))
            param_add = max(4, floor(0.002 * N))
            clique_indices = []
            not_clique_indices = []
            for i in range(N):
                if x_candidate[i] == 1:
                    clique_indices.append(i)
                else:
                    not_clique_indices.append(i)
            if len(clique_indices) > 0:
                if len(clique_indices) < param_rem:
                    param_rem = len(clique_indices)
                choice_remove = np.random.choice(
                    clique_indices, param_rem, replace=False)
                for i in choice_remove:
                    p_switch_remove = np.random.uniform()
                    if p_switch_remove < p:
                        x_candidate[i] = 0
                        not_clique_indices.append(i)
                        clique_indices = [j for j in clique_indices if j != i]
            if len(not_clique_indices) < param_add:
                param_add = len(not_clique_indices)
            choice = np.random.choice(
                not_clique_indices, param_add, replace=False)
            added = []
            for i in choice:
                if len(clique_indices) + len(added) < K and len([j for j in added if A[i, j] != 1]) == 0 and len([j for j in clique_indices if A[i, j] != 1]) == 0:
                    x_candidate[i] = 1
                    added.append(i)
            return x_candidate, time_complexity
        else:
            if N < k:
                k = N
            clique_indices1 = [i for i in range(N) if x_candidate[i] == 1]
            k_remove = max(
                1, floor(min(len(clique_indices1), k) * param_remove))
            k_add = k - k_remove
            if len(clique_indices1) == 0:
                # if the current clique is empty, then try to add a quarter of the target size
                k_add = floor(K * 0.25)
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
                    for i in np.random.choice(N, N, replace=False):
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
            if k_add > len(common_neighbors):
                k_add = len(common_neighbors)
            choice_add = np.random.choice(
                common_neighbors, k_add, replace=False)
            size_of_the_clique_before_adding = x_candidate.sum()
            added = []
            for i in choice_add:
                p_switch = np.random.uniform()
                p_accept = p
                if beta > 0:
                    p_accept += beta * 0.25
                if p_switch < p_accept:
                    if size_of_the_clique_before_adding + len(added) < K and len([j for j in added if A[i, j] != 1]) == 0:
                        x_candidate[i] = 1
                        added.append(i)
            return x_candidate, time_complexity
    return x_candidate, time_complexity


def metropolisHastings(A, N, K, x_init, n_steps, log_K_over_N, log_1_minus_K_over_N, N_count_constant, beta=1.0, param_k=10, print_progress=False, param_remove=0.5, A_neighbors=None, with_time_complexity=False):
    """
    Perform n_steps of the Metropolis Hastings algorithm

    Parameters:
    A, N, K, x_init, n_steps: in the context are self explanatory
    log_K_over_N, log_1_minus_K_over_N, N_count_constant: are constants for computation
    beta: the inverse temperature
    param_k: parameter k for the drawCandidate method (number of change's tries in the estimate)
    print_progress: bool, show or not a progress bar (when used in the PT algorithm, please set to False)
    param_remove: float in (0, 1) for the drawCandidate method
    A_neighbors: see createRandomGraphWithPlantedClique
    with_time_complexity: whether to compute the order of the operations needed to compute the common neighbors in the drawCandidate method

    Return:
    x, H_x, info
    x: np array representing the new estimate
    H_x: float, energy associated with the new estimate
    count_changes: # of time the candidate has been accepted
    if with_time_complexity is True: return also the order of the # of operations needed in the drawCandidate method
    """

    x = np.copy(x_init)
    H_x = H(x, N, log_K_over_N, log_1_minus_K_over_N, N_count_constant)
    count_changes = 0
    size_of_clique = 0
    if print_progress:
        printProgressBar(0, n_steps, prefix=f"Progress:",
                         suffix="Complete (size of clique estimate: 0)", length=50)
    time_complexity = 0
    for i in range(n_steps):
        if print_progress:
            printProgressBar(i + 1, n_steps, prefix=f"Progress:",
                             suffix=f"Complete (size of clique estimate: {size_of_clique})", length=50)
        p = 0.5
        candidate_method = "switch_k"
        x_candidate, step_time_complexity = drawCandidate(
            x, N, K, A, candidate_method, p=p, k=param_k, param_remove=param_remove, beta=beta, A_neighbors=A_neighbors)
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
            count_changes += 1
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
    if with_time_complexity:
        return x, H_x, count_changes, time_complexity
    return x, H_x, count_changes


def testMetropolis(configs):
    """
    configs: list of config for each test, a config is a dict with the keys "N", "K", "n_steps", "param_k"
    """
    for config in configs:
        print("===================== START test =====================")
        print("CONFIGURATION:")
        print("N", config["N"])
        print("K", config["K"])
        print("n_steps", config["n_steps"])
        print("param_k", config["param_k"])
        N, K, param_k = config["N"], config["K"], config["param_k"]
        A, v = createRandomGraphWithPlantedClique(N, K)
        truth = [i for i in range(N) if v[i] == 1]
        n_steps = config["n_steps"]
        log_K_over_N = np.log(K / N)
        log_1_minus_K_over_N = np.log(1 - K / N)
        N_count_constant = N * (N - 1)
        x_init = np.zeros(N)
        estimate2, H, count_changes = metropolisHastings(
            A, N, K, x_init, n_steps, log_K_over_N, log_1_minus_K_over_N, N_count_constant, param_k=param_k, print_progress=True)
        estimate_indices = [i for i in range(N) if estimate2[i] == 1]
        diff_not_in_truth = [i for i in estimate_indices if i not in truth]
        diff_not_in_estimate = [i for i in truth if i not in estimate_indices]
        try:
            print("Clique size in estimate:", len(estimate_indices))
            print("Count not in truth but in estimate:", len(diff_not_in_truth))
            print("Count not in estimate but in truth:",
                  len(diff_not_in_estimate))
        except:
            pass

        try:
            print("Changes (%):", float(count_changes) / n_steps * 100)
        except:
            pass
        print("===================== END test =====================")


def performMetropolisOnAllReplicas(estimates, betas, A, N, K, param_k, n_steps, log_K_over_N, log_1_minus_K_over_N, N_count_constant, param_remove=0.5, with_threading=False, A_neighbors=None):
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
                                       N_count_constant, betas[i], param_k, param_remove=param_remove, A_neighbors=A_neighbors, with_time_complexity=True) for i in range(len(betas))]
        for i, f in enumerate(futures):
            new_estimates[i], new_energies[i], monitoring[i], time_complexities[i] = f.result(
            )
    else:
        for i, beta in enumerate(betas):
            x, H_x, count_changes, time_complexity = metropolisHastings(
                A, N, K, estimates[i], n_steps, log_K_over_N, log_1_minus_K_over_N, N_count_constant, beta, param_k, param_remove=param_remove, A_neighbors=A_neighbors, with_time_complexity=True)
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
    # not sure whether change possibility for all or for pairs ?
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


def parallelTempering(A, N, K, betas, param_k, n_steps=5, switchConfig={"how": "consecutive", "reverse": False}, without_plot=True, param_remove=0.5, plot_file_name="sampling", A_neighbors=None, with_threading=False):
    """
    Perform the parallel tempering method with Metropolis Hastings steps

    Parameters:
    A: np matrix according to the paper (with variables A_ij = 1 if v[i]*v[j] == 1, else: 0 or 1 with probability 1/2 each)
    N: size of the graph
    K: size of the planted clique
    betas: list of inverse temperatures
    param_k: parameter for the drawCandidate of the Metropolis algorithm (how many elements are tried to be changed)
    n_steps: number of Metropolis steps for each replica before each configuration switch try
    switchConfig: see performSwitchConfiguration

    Return:
    x, monitoring_metropolis, monitoring_tempering, n_iterations
    x: the estimated clique
    monitoring_metropolis, monitoring_tempering: monitoring
    n_iterations: number of iterations done
    """
    start = datetime.now()

    log_K_over_N = np.log(K / N)  # constant for computation
    log_1_minus_K_over_N = np.log(1 - K / N)  # constant for computation
    N_count_constant = N * (N - 1)  # constant for computation

    control = 0  # current number of iteration of the algorithm
    # maximum number of iterations of the algorithm
    limit = 100000000 if N != 4000 else 1000000000  # prevent infinite search

    # initialization of the estimates for each replica
    estimates = [np.zeros(N) for i in range(len(betas))]

    # initialization of the energies of the estimates for each replica
    energies = [0.0 for i in range(len(betas))]

    # keep track of metropolis changes acceptance for each replica
    monitoring_metropolis = [0 for _ in range(len(betas))]

    # keep track of the current total number of switches of configurations
    current_number_changes_temp = 0
    current_number_changes_temp_beta_1 = 0

    # keep track of the current estimated clique size in time
    size_evolution_in_time = [0]
    size_estimate_clique = 0  # current estimated clique size

    avg_time_complexity = 0

    # initialize the progress bar indicating the percentage of the current estimated clique size against K
    printProgressBar(0, K, prefix=f"Progress:",
                     suffix=f"of the clique size (step #{control}, beta config flips: {current_number_changes_temp})", length=50)
    # run the algorithm
    while control < limit and size_estimate_clique < K:
        # perform Metropolis on all replicas
        estimates, energies, new_monitoring_metropolis, step_avg_time_complexity = performMetropolisOnAllReplicas(
            estimates, betas, A, N, K, param_k, n_steps, log_K_over_N, log_1_minus_K_over_N, N_count_constant, param_remove=param_remove, A_neighbors=A_neighbors, with_threading=with_threading)
        # monitor changes by Metropolis
        monitoring_metropolis = [(control * monitoring_metropolis[i] + float(
            new_monitoring_metropolis[i]) / n_steps) / (control + 1) for i in range(len(monitoring_metropolis))]

        avg_time_complexity += step_avg_time_complexity

        # perform configurations
        estimates, energies, monitoring_tempering_step = performSwitchConfiguration(
            estimates, energies, betas, switchConfig)

        # keep track of the configurations switches
        current_number_changes_temp += monitoring_tempering_step["switchCount"]
        current_number_changes_temp_beta_1 += monitoring_tempering_step["switchBeta1"]

        # size of the current estimated clique
        size_estimate_clique = estimates[0].sum()
        if not without_plot:
            size_evolution_in_time.append(size_estimate_clique)

        # update progress bar
        change_accept_beta_1 = round(
            monitoring_metropolis[0] * 100, 1)
        change_accept_beta_0_55 = round(
            monitoring_metropolis[9] * 100, 1)
        change_accept_beta_0_1 = round(
            monitoring_metropolis[18] * 100, 1)
        printProgressBar(size_estimate_clique, K, prefix=f"Progress:",
                         suffix=f"of the clique size (step #{control + 1}, beta config flips: {current_number_changes_temp}, accept: | 1.0: {change_accept_beta_1}%, 0.55: {change_accept_beta_0_55}%, 0.1: {change_accept_beta_0_1}%)", length=50)

        # an iteration was done
        control += 1

    avg_time_complexity = (avg_time_complexity * len(betas)) / control

    # the clique has not been recovered inside the limit
    if size_estimate_clique != K:
        printProgressBar(K, K, prefix=f"Progress:",
                         suffix=f"of the clique size (step #{control + 1}, beta config flips: {current_number_changes_temp})", length=50)
    if control >= limit:
        print("Failed to recover")

    stop = datetime.now()

    # plot evolution of the current estimated clique size in time
    if not without_plot:
        plt.plot(np.arange(len(size_evolution_in_time)),
                 np.array(size_evolution_in_time))
        plt.savefig(f"{plot_file_name}.png")
        plt.close("all")

    return estimates[0], monitoring_metropolis, {"switchCount": current_number_changes_temp, "switchCountBeta1": current_number_changes_temp_beta_1}, {"iterations": control, "time": (stop - start).seconds, "avgTimeComplexity": avg_time_complexity}


def testParallelTempering(N, K, betas=[], n_steps=5, from_file=""):
    """
    Test the parallel tempering algorithm
    """
    # print test settings
    print("===================== START test =====================")
    print("SETTINGS:")
    print("N", N)
    print("K", K)
    print("param_k", param_k)

    # create planted random graph
    if len(from_file) > 0:
        A, v = openGraphAndClique(from_file)
        A_neighbors = None
    else:
        A, v, A_neighbors = createRandomGraphWithPlantedClique(
            N, K, with_neighbors=True)
    truth = [i for i in range(N) if v[i] == 1]

    # initialize inverse temperatures as in the paper
    if len(betas) == 0:
        betas = [1 - i * 0.05 for i in range(19)]

    # parameters
    param_remove = getParam_Remove(N, K)
    param_k = getParam_k(N, K)

    # run PT and compute elapsed time
    estimate, monitoring_metropolis, monitoring_tempering, time_result = parallelTempering(
        A, N, K, betas, param_k, n_steps, A_neighbors=A_neighbors, with_threading=True, param_remove=param_remove)

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
    if K / np.log2(N) > 3:
        return min(60, max(1, floor(K * 0.5))) + floor(N / 1000.0)
    return max(1, 2 * K)


def getParam_Remove(N, K):
    if K / np.log2(N) > 3:
        return 0.25
    return 0.5


def generateGraphSamplesAndSave(n_samples, N, K, filename_base="graph"):
    for i in range(1, n_samples + 1):
        np.random.seed(i * 100)
        A, v = createRandomGraphWithPlantedClique(N, K)
        f_name = f"{filename_base}_N{N}_K{K}_{i * 100}"
        saveGraphAndClique(A, v, f_name)
        print("Created file:", f_name + ".npy")
    print("Done!")


def saveGraphAndClique(A, v, filename):
    with open(f"{filename}.npy", "wb") as f:
        np.save(f, A)
        np.save(f, v)


def openGraphAndClique(filename):
    with open(f"{filename}.npy", "rb") as f:
        A = np.load(f)
        v = np.load(f)
        return A, v


# data for the time convergence changing K analyze
K_tildes_N2000 = [1.64, 1.73, 1.82, 1.92, 2.01,
                  2.09, 2.18, 2.28, 2.37, 2.46, 2.55, 2.64, 2.73]
K_tildes_N3000 = [1.73, 1.82, 1.9, 1.99, 2.07, 2.16,
                  2.25, 2.34, 2.43, 2.51, 2.6, 2.68, 2.77, 2.86]
K_tildes_N4000 = [1.92, 2.01, 2.09, 2.17, 2.26, 2.34, 2.42,
                  2.51, 2.59, 2.67, 2.76, 2.84, 2.92, 3.01, 3.09, 3.17, 3.26]
K_tildes_N5000 = [2.19, 2.28, 2.36, 2.44, 2.52, 2.61, 2.68,
                  2.77, 2.85, 2.93, 3.01, 3.09, 3.17, 3.26, 3.34, 3.42, 3.5]


def timeOfConvergenceChangingN(Ns, n_samples, K_to_N_factor=0.125, K_tilde=-1):
    """
    Run the PT algorithm for each N in Ns (n_samples for each N) and save # of iterations needed for the PT in files

    Parameters:
    Ns: list of integers (sizes of graphs to be sampled)
    n_samples: the number of graph's realizations for each N in Ns
    K_to_N_factor: if K_tilde < 0 then the size of the planted clique is N * K_to_N_factor
    K_tilde: compute the size of the planted clique as K_tilde * log_2(N) 
    """
    results = [[] for _ in range(len(
        Ns))]  # save the results (number of PT iterations needed to recover the planted clique)
    betas = [1 - i * 0.05 for i in range(19)]  # default betas
    n_steps = 5  # Metropolis steps
    for i, N in enumerate(Ns):
        avg_time_complexity = 0
        if K_tilde > 0:
            K = getKFromKTilde(N, K_tilde)
        else:
            K = floor(K_to_N_factor * N)
        param_k = getParam_k(N, K)
        print("===================== START sampling =====================")
        print("SETTINGS:")
        print("N", N)
        print("K", K)
        print("param_k", param_k)
        samples_done_count = 0
        param_remove = getParam_Remove(N, K)
        while samples_done_count < n_samples:
            A, v, A_neighbors = createRandomGraphWithPlantedClique(
                N, K, with_neighbors=True)
            truth = [i for i in range(N) if v[i] == 1]
            estimate, monitoring_metropolis, monitoring_tempering, time_res = parallelTempering(
                A, N, K, betas, param_k, n_steps, without_plot=True, A_neighbors=A_neighbors, param_remove=param_remove, with_threading=True)
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
        print(f"Sampling for N={N} finished with time complexity:",
              round(avg_time_complexity / samples_done_count))
        with open(f"intermediate_results_time_of_convergence_changing_N_{N}.npy", "wb") as f:
            np.save(f, np.array([x for x in results if len(x) > 0]))
    filename_suffix = datetime.now().isoformat()
    filename_suffix = filename_suffix.replace(":", "-")
    if "." in filename_suffix:
        filename_suffix = filename_suffix[:filename_suffix.index(".")]
        filename_suffix = filename_suffix.replace(".", "-")
    with open(f"final_results_time_of_convergence_changing_N_from_{Ns[0]}_to_{Ns[-1]}_{filename_suffix}.npy", "wb") as f:
        np.save(f, np.array(results))


def timeOfConvergenceChangingK(N_param=0, n_samples=1, send_result_email=False):
    """
    Run the PT algorithm for each N in Ns (or only for N_param if not 0) and over the different K_tilde in the corresponding list

    Parameters:
    N_param: if 0 then run over all N in Ns, else if N_param is in Ns then run only for the specified N_param
    n_samples: number of samples for each K_tilde
    """
    pwd = ""
    if send_result_email:
        pwd = input("Input password and press Enter: ")
    n_realizations_per_point = n_samples
    Ns = [2000, 3000, 4000, 5000, 1000]
    K_tildes_N2000.reverse()
    K_tildes_N3000.reverse()
    K_tildes_N4000.reverse()
    K_tildes_N5000.reverse()
    K_tildes = [K_tildes_N2000, K_tildes_N3000,
                K_tildes_N4000, K_tildes_N5000, K_tildes_N2000]
    results = {"2000": [], "3000": [], "4000": [], "5000": [], "1000": []}
    betas = [1 - i * 0.05 for i in range(19)]
    n_steps = 5  # Metropolis steps
    for i, N in enumerate(Ns):
        if N_param != 0 and N_param != N:
            continue
        for j, K_tilde in enumerate(K_tildes[i]):
            results[str(N)].append(
                [0 for _ in range(n_realizations_per_point)])
            K = getKFromKTilde(N, K_tilde)
            param_k = getParam_k(N, K)
            param_remove = getParam_Remove(N, K)
            print("===================== START sampling =====================")
            print("SETTINGS:")
            print("N", N)
            print("K", K)
            print("K_tilde", K_tilde)
            print("param_k", param_k)
            realizations_done_count = 0
            while realizations_done_count < n_realizations_per_point:
                A, v, A_neighbors = createRandomGraphWithPlantedClique(
                    N, K, with_neighbors=True)
                truth = [i for i in range(N) if v[i] == 1]
                estimate, monitoring_metropolis, monitoring_tempering, time_res = parallelTempering(
                    A, N, K, betas, param_k, n_steps, without_plot=True, A_neighbors=A_neighbors, param_remove=param_remove, with_threading=True)
                estimate_indices = [i for i in range(N) if estimate[i] == 1]
                diff_not_in_truth = [
                    i for i in estimate_indices if i not in truth]
                diff_not_in_estimate = [
                    i for i in truth if i not in estimate_indices]
                if len(diff_not_in_truth) == 0 and len(diff_not_in_estimate) == 0:
                    results[str(
                        N)][j][realizations_done_count] = time_res["iterations"]
                    realizations_done_count += 1
                    print(
                        f"Clique {realizations_done_count} recovered (N: {N}, K_tilde: {K_tilde})")
                # with open(f"results_time_of_convergence_keep_{N}.npy", "wb") as f:
                #     np.save(f, np.array(results[str(N)]))
        with open(f"results_time_of_convergence_{N}_end.npy", "wb") as f:
            np.save(f, np.array(results[str(N)]))
        if N_param != 0 and send_result_email:
            status = send_email("yan-lhomme@outlook.com", "Test time of convergence",
                                f"Test for {n_realizations_per_point} realizations", f"results_time_of_convergence_{N}_end.npy", pwd)
            print("Sent email:", status)
    if N_param == 0:
        with open(f"results_time_of_convergence_finished.npy", "wb") as f:
            np.save(f, np.array(results["2000"]))
            np.save(f, np.array(results["3000"]))
            np.save(f, np.array(results["4000"]))
            np.save(f, np.array(results["5000"]))
            if 1000 in Ns:
                np.save(f, np.array(results["1000"]))
        if send_result_email:
            status = send_email("yan-lhomme@outlook.com", "Test time of convergence",
                                f"Test for {n_realizations_per_point} realizations", "results_time_of_convergence_finished.npy", pwd)
            print("Sent email:", status)


# Plots

def createGraphTimeOfConvergence(filename):
    # warning reverse
    with open(f"{filename}.npy", "rb") as f:
        iterations = np.load(f)
    iterations_mean = [x.mean() for x in iterations]
    iterations_std_dev = [x.std() for x in iterations]
    # iterations_mean.reverse()
    plt.errorbar(np.array(K_tildes_N2000), np.array(iterations_mean),
                 yerr=np.array(iterations_std_dev), fmt=".k")
    # plt.yscale("log")
    plt.show()


def createGraphTimeOfConvergencePTStepsChangingN(Ns, filename, K_computation_display, param_k_display):
    with open(f"{filename}.npy", "rb") as f:
        iterations = np.load(f)
    iterations_mean = [x.mean() for x in iterations]
    iterations_std_dev = [x.std() for x in iterations]
    # interpolate_line_x = [Ns[0], Ns[-1]]
    # interpolate_line_y = np.interp(interpolate_line_x, Ns, iterations_mean)
    plt.errorbar(np.array(Ns), np.array(iterations_mean),
                 yerr=np.array(iterations_std_dev), fmt="xb", ecolor="b", capsize=3)
    label = f"Size of the graph (N) with planted clique of size K {K_computation_display}"
    if len(param_k_display) > 0:
        label += f"and parameter k {param_k_display} (for candidate)"
    plt.xlabel(label)
    # plt.plot(np.array(interpolate_line_x),
    #          np.array(interpolate_line_y), color="c")
    plt.ylabel("Number of PT steps needed")
    plt.show()


def createGraphTimeOfConvergencePseudoTimeChangingN(Ns, filename, filename_time_complexity, K_computation_display, param_k_display):
    with open(f"{filename}.npy", "rb") as f:
        iterations = np.load(f)
    with open(f"{filename_time_complexity}.npy", "rb") as f:
        times_complexities = np.load(f)
    if len(times_complexities) != len(Ns):
        new_times_complexities = []
        for i in range(len(Ns)):
            new_times_complexities.append(times_complexities[i])
        times_complexities = np.array(new_times_complexities)

    times_complexities_standardized = [
        times_complexities[i] / Ns[i] for i in range(len(times_complexities))]
    iterations_with_time_complexity_mean = [np.array(
        [x[j] * times_complexities_standardized[i] for j in range(len(x))]).mean() for i, x in enumerate(iterations)]
    iterations_with_time_complexity_std_dev = [np.array(
        [x[j] * times_complexities_standardized[i] for j in range(len(x))]).std() for i, x in enumerate(iterations)]
    # interpolate_line_x = [Ns[0], Ns[-1]]
    # interpolate_line_y = np.interp(
    #     interpolate_line_x, Ns, iterations_with_time_complexity_mean)
    plt.errorbar(np.array(Ns), np.array(iterations_with_time_complexity_mean),
                 fmt="xb", ecolor="b", capsize=3)  # yerr=np.array(iterations_with_time_complexity_std_dev)
    # plt.plot(np.array(interpolate_line_x),
    #          np.array(interpolate_line_y), color="c")
    label = f"Size of the graph (N) with planted clique of size K {K_computation_display}"
    if len(param_k_display) > 0:
        label += f"and parameter k {param_k_display} (for candidate)"
    plt.xlabel(label)
    plt.ylabel("'Time': # of PT steps * ((avg # of operations for N) / N)")
    plt.show()


def createGraphPseudoTimeOperationsChangingN(Ns, filename_time_complexity, K_computation_display, param_k_display):
    with open(f"{filename_time_complexity}.npy", "rb") as f:
        times_complexities = np.load(f)
    if len(times_complexities) != len(Ns):
        new_times_complexities = []
        for i in range(len(Ns)):
            new_times_complexities.append(times_complexities[i])
        times_complexities = np.array(new_times_complexities)
    times_complexities_standardized = [
        times_complexities[i] / Ns[i] for i in range(len(times_complexities))]
    interpolate_line_x = [Ns[0], Ns[-1]]
    interpolate_line_y = np.interp(
        interpolate_line_x, Ns, times_complexities_standardized)
    plt.errorbar(np.array(Ns), np.array(
        times_complexities_standardized), fmt="xb", ecolor="b", capsize=3)
    plt.plot(np.array(interpolate_line_x),
             np.array(interpolate_line_y), color="k")
    label = f"Size of the graph (N) with planted clique of size K {K_computation_display}"
    if len(param_k_display) > 0:
        label += f"and parameter k {param_k_display} (for candidate)"
    plt.xlabel(label)
    plt.ylabel("(Avg # of operations for N per PT step) / N")
    plt.show()


if __name__ == '__main__':
    # ===============================
    # To test the clique recovery with PT uncomment this section and change the values of K_tilde_test, N_test, K_test (and param_k_test) according to your needs
    # K_tilde_test = 4
    # N_test = 2000
    # K_test = getKFromKTilde(N_test, K_tilde_test)
    # param_k_test = 2 * K_test
    # testParallelTempering(N_test, K_test, param_k_test)
    # ===============================

    # ===============================
    # To sample the convergence of PT with the values of the paper (changing K) uncomment this section
    timeOfConvergenceChangingK(
        N_param=3000, n_samples=5, send_result_email=True)
    # ===============================

    # ===============================
    # To sample the convergence of PT by changing N uncomment this section
    # Ns = [200, 500, 1000, 2000, 3000, 4000,
    #       5000, 10000, 20000]  # the N's to be sampled
    # Ns = [200, 500, 1000, 2000, 3000, 4000, 5000]  # the N's to be sampled
    # # size of the clique K will be floor(N * K_to_N_factor)
    # K_to_N_factor = 0.125
    # K_tilde = 6
    # n_samples = 5  # number of graph realizations per N
    # timeOfConvergenceChangingN(Ns, n_samples, K_to_N_factor, K_tilde)
    # ===============================

    # ===============================
    # Create plots from data
    # Ns = [200, 500, 1000, 2000, 3000, 4000,
    #       5000]  # the N's to be sampled
    # filename = f"final_results_time_of_convergence_changing_N_from_{Ns[0]}_to_{Ns[-1]}_2022-05-06T22-38-06"
    # filename_times = f"time_complexity_from_200_to_20000_Kfactor_0_125"
    # createGraphPseudoTimeOperationsChangingN(
    #     Ns, filename_times, "(6 * log_2(N))", "")
    # createGraphTimeOfConvergencePTStepsChangingN(
    #     Ns, filename, "(6 * log_2(N))", "")
    # createGraphTimeOfConvergencePseudoTimeChangingN(
    #     Ns, filename, filename_times, "(6 * log_2(N))", "")
    # ===============================
    pass


"""
Old functions

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

from math import ceil, floor
import numpy as np
import concurrent.futures
from progressbar import printProgressBar
from datetime import datetime
from mails import send_email


PROBLEM_SPECIFIC_VALUE = 0


def createRandomGraphWithPlantedClique(N, K, d, with_neighbors=False):
    """
    Return matrix A and planted clique vector v of a realization of a planted random graph of size N and planted clique size K
    if with_neighbors is True then it returns also a list of list of neighbors indices for each node
    """
    rho = float(K) / N
    c_in = d * (1 - 2 * rho) / (N * (1 - rho) * (1 - rho))
    c_out = d / (N * (1 - rho))
    v_index_choices = np.random.choice(N, K, replace=False)
    v = np.array(
        [1 if i in v_index_choices else 0 for i in range(N)], dtype=np.int8)
    A = np.zeros((N, N), dtype=np.int8)
    A_neighbors = [[] for _ in range(N)]
    for i in range(N-1):
        for j in range(i + 1, N):
            add_edge_probability = 0
            if v[i] == 0 and v[j] == 0:
                add_edge_probability = c_in
            elif (v[i] == 1 or v[j] == 1) and v[i] * v[j] == 0:
                add_edge_probability = c_out
            p = np.random.uniform()
            if p < add_edge_probability:
                A[i, j] = 1
                A_neighbors[i].append(j)
                A_neighbors[j].append(i)
    A = A + A.T
    if with_neighbors:
        A_non_neighbors = []
        for neighbors in A_neighbors:
            A_non_neighbors.append([j for j in range(N) if j not in neighbors])
        print("Average degree:", np.array(
            [len(x) for x in A_neighbors]).mean())
        return A, v, A_non_neighbors
    return A, v


def H(x, A):
    # indices = [i for i in range(len(x)) if x[i] == 1]
    # result = 0
    # for i in indices:
    #     for j in [k for k in indices if k > i]:
    #         if A[i, j] == 1:
    #             result += 1
    M = np.outer(x, x)
    C = np.multiply(A, M)
    return round(0.5 * C.sum())


def drawCandidate(x):
    elements_of_IS = []
    non_elements_of_IS = []
    for i in range(len(x)):
        if x[i] == 1:
            elements_of_IS.append(i)
        else:
            non_elements_of_IS.append(i)
    change_index = np.random.choice(elements_of_IS, 1)
    index_new = np.random.choice(non_elements_of_IS, 1)
    x_candidate = np.copy(x)
    x_candidate[change_index] = 0
    x_candidate[index_new] = 1
    return x_candidate


def MC(x_init, A, n_steps, beta):
    x = np.copy(x_init)
    H_x = H(x_init, A)
    count_changes = 0
    for _ in range(n_steps):
        x_candidate = drawCandidate(x)
        H_candidate = H(x_candidate, A)
        if H_candidate == float("inf"):
            # this should not happen
            continue
        elif H_x == float("inf"):
            # this should not happen
            count_changes += 1
            x = x_candidate
            H_x = H_candidate
            continue
        if H_candidate <= H_x:
            count_changes += 1
            x = x_candidate
            H_x = H_candidate
        else:
            alpha = np.exp(-beta * (H_candidate - H_x))
            p_accept = np.random.uniform()
            if p_accept < alpha:
                count_changes += 1
                x = x_candidate
                H_x = H_candidate
    return x, H_x, count_changes


def performMetropolisOnAllReplicas(estimates, betas, A, N, n_steps):
    """
    Call the Metropolis algorithm for each replica

    Parameters: self explanatory in the context

    Return:
    new_estimates: list of np arrays representing the new estimates for each replica
    new_energies: list of float of the energies associated to the new estimates
    monitoring: list of info (see metropolisHastings) for each replica
    """
    new_estimates = [np.zeros(N) for _ in range(len(betas))]
    new_energies = [0 for _ in range(len(betas))]
    monitoring = [0 for _ in range(len(betas))]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(
            MC, estimates[i], A, n_steps, betas[i]) for i in range(len(betas))]
    for i, f in enumerate(futures):
        new_estimates[i], new_energies[i], monitoring[i] = f.result(
        )

    return new_estimates, new_energies, monitoring


def performSwitchConfiguration(estimates, energies, betas, config={"how": "consecutive", "reverse": False}):
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
                new_estimates_i = np.copy(new_estimates[i])
                new_estimates[i] = np.copy(new_estimates[i + 1])
                new_estimates[i + 1] = new_estimates_i
                new_energies_i = new_energies[i]
                new_energies[i] = new_energies[i + 1]
                new_energies[i + 1] = new_energies_i
    return new_estimates, new_energies


def parallelTempering(A, N, K, betas, n_steps=5):
    """
    Perform the parallel tempering method with Metropolis Hastings steps

    Parameters:
    A: np matrix according to the paper (with variables A_ij = 1 if v[i]*v[j] == 1, else: 0 or 1 with probability 1/2 each)
    N: size of the graph
    K: size of the planted clique
    betas: list of inverse temperatures
    n_steps: number of Metropolis steps for each replica before each configuration switch try

    Return:
    x, iterations
    x: the estimated clique
    iterations: number of iterations done
    """
    K_constant = floor(K * (K - 1) * 0.5)
    betas_index_middle = floor(len(betas) * 0.5)  # used for monitoring

    control = 0  # current number of iteration of the algorithm
    # maximum number of iterations of the algorithm
    limit = 2000000  # prevent infinite search

    # initialization of the estimates for each replica
    estimates = [np.zeros(N) for _ in range(len(betas))]
    for b in range(len(betas)):
        init_indices = np.random.choice(N, K, replace=False)
        for i in init_indices:
            estimates[b][i] = 1

    # initialization of the energies of the estimates for each replica
    energies = [K_constant for _ in range(len(betas))]

    # keep track of metropolis changes acceptance for each replica
    monitoring_metropolis = [0 for _ in range(len(betas))]

    # initialize the progress bar indicating the percentage of the current estimated clique size against K
    printProgressBar(0, K_constant, prefix=f"Progress:",
                     suffix=f"of the clique size (step #{control})", length=20)
    # run the algorithm
    while control < limit and round(energies[0]) > 0:
        # perform Metropolis on all replicas
        estimates, energies, new_monitoring_metropolis = performMetropolisOnAllReplicas(
            estimates, betas, A, N, n_steps)
        # monitor changes by Metropolis
        monitoring_metropolis = [(control * monitoring_metropolis[i] + (float(new_monitoring_metropolis[i]) / n_steps)) / (
            control + 1) for i in range(len(monitoring_metropolis))]

        # perform configurations
        estimates, energies = performSwitchConfiguration(
            estimates, energies, betas)

        # update progress bar
        change_accept_beta_1 = round(
            monitoring_metropolis[0] * 100, 1)
        change_accept_beta_0_55 = round(
            monitoring_metropolis[betas_index_middle] * 100, 1)
        change_accept_beta_0_1 = round(
            monitoring_metropolis[-1] * 100, 1)
        printProgressBar(K_constant - round(energies[0]), K_constant, prefix=f"Progress:",
                         suffix=f"of the clique size (step #{control + 1}, accept: | 1.0: {change_accept_beta_1}%, {round(betas[betas_index_middle], 2)}: {change_accept_beta_0_55}%, {round(betas[-1], 2)}: {change_accept_beta_0_1}%)", length=20)
        # an iteration was done
        control += 1

    # the clique has not been recovered inside the limit
    if energies[0] > 0:
        printProgressBar(K_constant, K_constant, prefix=f"Progress:",
                         suffix=f"of the clique size (step #{control + 1})", length=20)
        print("Failed to recover")

    return estimates[0], control


def checkIfClique(x, A):
    """
    Check whether an estimate is a clique or not
    """
    for i in range(len(x)):
        if x[i] == 1:
            for j in range(len(x)):
                if j != i and x[j] == 1 and A[i, j] != PROBLEM_SPECIFIC_VALUE:
                    return False
    return True


def getKFromKTilde(N, K_tilde):
    """
    Return the size of the clique for a given K_tilde
    """
    return max(1, round(K_tilde * np.log2(N)))


def getKFromRho(N, rho):
    """
    Return the size of the IS for a given rho (density)
    """
    return max(1, round(rho * N))


def timeOfConvergenceChangingK(Ns, K_tildes, d=-1, n_samples=1, accept_other_clique=False, send_result_email=False):
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
    betas = [1 - i * 0.11 for i in range(7)]
    n_steps = 5  # Metropolis steps
    for i, N in enumerate(Ns):
        results.append([])
        for j, K_tilde in enumerate(K_tildes[i]):
            results[i].append([])
            if PROBLEM_SPECIFIC_VALUE == 0:
                K = getKFromRho(N, K_tilde)
            else:
                K = getKFromKTilde(N, K_tilde)
            print("===================== START sampling =====================")
            print("SETTINGS:")
            print("N", N)
            print("K", K)
            if PROBLEM_SPECIFIC_VALUE == 0:
                print("rho", K_tilde)
                print("d", d)
            else:
                print("K_tilde", K_tilde)
            realizations_done_count = 0
            while realizations_done_count < n_samples:
                A, v = createRandomGraphWithPlantedClique(N, K, d)
                truth = [i for i in range(N) if v[i] == 1]
                estimate, time_res = parallelTempering(
                    A, N, K, betas, n_steps=n_steps)
                estimate_indices = [i for i in range(N) if estimate[i] == 1]
                diff_not_in_truth = [
                    i for i in estimate_indices if i not in truth]
                diff_not_in_estimate = [
                    i for i in truth if i not in estimate_indices]
                if len(diff_not_in_truth) == 0 and len(diff_not_in_estimate) == 0:
                    results[i][j].append(time_res)
                    realizations_done_count += 1
                    print(
                        f"IS {realizations_done_count} recovered (N: {N}, K_tilde: {K_tilde})")
                else:
                    if accept_other_clique and checkIfClique(estimate, A) and len(estimate_indices) == K:
                        results[i][j].append(time_res)
                        realizations_done_count += 1
                        print(
                            f"Other IS {realizations_done_count} recovered (N: {N}, K_tilde: {K_tilde})")
                    else:
                        estimate_is_clique = checkIfClique(estimate, A)
                        print("WARNING: planted clique not recovered")
                        print("Estimate is clique:",
                              estimate_is_clique)
                        print(f"Estimate has size K = {K} ?",
                              len(estimate_indices))
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
    with open(f"IS_final_results_time_of_convergence_changing_N_multipleK_from_{Ns[0]}_to_{Ns[-1]}_{n_samples}samples_{filename_suffix}.npy", "wb") as f:
        for i in range(len(Ns)):
            np.save(f, np.array(results[i]))
        np.save(f, np.array(Ns))
        np.save(f, np.array(K_tildes))
    print(
        "Saved:", f"IS_final_results_time_of_convergence_changing_N_multipleK_from_{Ns[0]}_to_{Ns[-1]}_{n_samples}samples_{filename_suffix}.npy")
    if send_result_email:
        status = send_email("yan-lhomme@outlook.com", "Results time of convergence",
                            f"IS_final_results_time_of_convergence_changing_N_multipleK_from_{Ns[0]}_to_{Ns[-1]}_{n_samples}samples_{filename_suffix}.npy", pwd)
        print("Sent email:", status)


if __name__ == '__main__':
    Ns = [700]
    rhos = [[0.14, 0.13] for N in Ns]
    d = 40.0
    n_samples = 5  # number of graph realizations per N
    timeOfConvergenceChangingK(
        Ns, rhos, d, n_samples=n_samples)
    pass

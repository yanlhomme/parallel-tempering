from datetime import datetime
from math import ceil, floor
import numpy as np

from progressbar import printProgressBar

# np.random.seed(300)


def createRandomGraphWithPlantedClique(N, K):
    v_index_choices = np.random.choice(N, K, replace=False)
    v = np.array([1 if i in v_index_choices else 0 for i in range(N)])
    A = np.zeros((N, N))
    for i in range(N-1):
        for j in range(i + 1, N):
            if v[i] * v[j] == 1:
                A[i, j] = 1
            else:
                p = np.random.uniform()
                if p < 0.5:
                    A[i, j] = 1
    A = A + A.T
    return A, v


# def betheFreeEnergy(N, z, z_ij):
#     return - 1 / N * (np.log(z).sum() - (np.array([np.log(np.array([z_ij[i, j] for j in range(N) if j != i])).sum() for i in range(N)])).sum())


def P_local_prior(x_i, K, N):
    # return (K / N)**x_i * (1 - K / N)**(1-x_i)
    if x_i == 1:
        return K / N
    return 1 - K / N


def preventAbsenceOfLinksBetweenTwoElmentsOfTheClique(A, x):
    for i in range(A.shape[0] - 1):
        for j in range(i + 1, A.shape[1]):
            if A[i, j] == 0 and x[i]*x[j] == 1:
                return True
    return False


def H(x, A, K, N):
    if preventAbsenceOfLinksBetweenTwoElmentsOfTheClique(A, x):
        return float("inf")
    second_sum = 0
    for i in range(N-1):
        for j in range(i + 1, N):
            if A[i, j] == 1:
                # does the first term of the sum vanishes when A_ij == 1 ?
                second_sum += np.log((1 + x[i] * x[j]) / 2)
            else:
                second_sum += np.log((1 - x[i] * x[j]) / 2)
    number_ones = x.sum()
    first_sum = number_ones * np.log(K / N)
    first_sum += (N - number_ones) * np.log(1 - K / N)
    # return np.log(np.array([P_local_prior(x_i, K, N) for x_i in x])).sum() - second_sum
    return first_sum - second_sum


def drawCandidate(x, N, method="switch_1", p=0.5, k=1):
    """
    methods:
        switch_1: switch 1 element of x with probability p
        switch_k: switch k element of x independently with probability p
    """
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
    """
    methods:
        switch_1: switch 1 element of x with probability p
        switch_k: switch k element of x independently with probability p
    """
    x_candidate = np.copy(x)
    if method in ["switch_1", "switch_k"]:
        if method == "switch_1":
            k = 1
        available_elements_for_change = [i for i in range(N) if x_candidate[i] == 1 or (
            x_candidate[i] == 0 and len([j for j in range(N) if j != i and x_candidate[j] == 1 and A[i, j] != 1]) == 0)]
        if len(available_elements_for_change) < k:
            k = len(available_elements_for_change)
        choice = np.random.choice(
            available_elements_for_change, k, replace=True)
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


def drawCandidateWithCliqueCheckAndPenalizeRemoving(x, N, A, method="switch_1", p=0.5, k=1):
    """
    methods:
        switch_1: switch 1 element of x with probability p
        switch_k: switch k element of x independently with probability p
    """
    x_candidate = np.copy(x)
    if method in ["switch_1", "switch_k"]:
        if method == "switch_1":
            k = 1
        available_elements_for_change = [i for i in range(N) if x_candidate[i] == 1 or (
            x_candidate[i] == 0 and len([j for j in range(N) if j != i and x_candidate[j] == 1 and A[i, j] != 1]) == 0)]
        choice = np.random.choice(
            available_elements_for_change, k, replace=True)
        for i in choice:
            p_switch = np.random.uniform()
            if x_candidate[i] == 1:
                if p_switch < p / 2:
                    x_candidate[i] = 0
            else:
                if p_switch < p:
                    if len([j for j in range(N) if j != i and x_candidate[j] == 1 and A[i, j] != 1]) == 0:
                        x_candidate[i] = 1
        return x_candidate
    return x_candidate


def metropolisHastingsAroundSolutionInit(A, N, K, n_steps, v, n_changes):
    x_init = np.copy(v)
    choice = np.random.choice(N, n_changes, replace=False)
    for i in choice:
        x_init[i] = 1 - x_init[i]
    return metropolisHastings(A, N, K, x_init, n_steps)


def metropolisHastingsRandomInit(A, N, K, n_steps, param_k=10):
    # x_init = np.random.choice([0, 1], N, replace=True, p=[
    #                           1 - float(K) / N, float(K) / N])
    # control = 0
    # limit = 10000
    # while control < limit and preventAbsenceOfLinksBetweenTwoElmentsOfTheClique(A, x_init):
    #     x_init = np.random.choice([0, 1], N, replace=True, p=[
    #                           1 - float(K) / N, float(K) / N])
    #     control += 1
    # if control >= limit:
    #     print("Problem initialization !!!!!")
    x_init = np.zeros(N)
    return metropolisHastings(A, N, K, x_init, n_steps, param_k=param_k)


def metropolisHastings(A, N, K, x_init, n_steps, beta=1.0, param_k=10):
    x = np.copy(x_init)
    H_x = H(x, A, K, N)
    count_H_candidate_inf = 0
    count_H_x_inf = 0
    count_changes = 0
    alphas = []
    size_of_clique = 0
    # printProgressBar(0, n_steps, prefix=f"Progress:",
    #                  suffix="Complete (size of clique estimate: 0)", length=50)
    # avg_H_time = 0
    for i in range(n_steps):
        # printProgressBar(i + 1, n_steps, prefix=f"Progress:",
        #                  suffix=f"Complete (size of clique estimate: {size_of_clique})", length=50)
        x_candidate = drawCandidateWithCliqueCheck(
            x, N, A, "switch_k", k=param_k)
        #print([k for k in range(N) if x_candidate[k] == 1])
        start = datetime.now()
        H_candidate = H(x_candidate, A, K, N)
        stop = datetime.now()
        # avg_H_time += (stop - start).microseconds / 1000.0
        if H_candidate == float("inf"):  # and H_x != float("inf"):
            count_H_candidate_inf += 1
            continue
        elif H_x == float("inf"):
            count_H_x_inf += 1
            count_changes += 1
            x = x_candidate
            H_x = H_candidate
            size_of_clique = x.sum()
            continue
        # alpha = np.exp(-beta * (H_candidate - H_x))
        # alphas.append(alpha)
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
    #alpha_avg = np.average(np.array(alphas))
    # alpha_1 = len([x for x in alphas if x == 1.0])
    # "alphaAvg": alpha_avg,"alpha1": alpha_1
    # if beta == 1.0:
    #     avg_H_time /= n_steps
    #     print("Avg time compute H:", avg_H_time)
    return x, H_x, {"changes": count_changes, "candInf": count_H_candidate_inf, "xInf": count_H_x_inf}


# def figure1_stabilityOfPlantedSolution(sizes, Ks_tilde, n_samples):
#     results = np.zeros((len(sizes), len(Ks_tilde)))
#     for i, N in enumerate(sizes):
#         for j, K_tilde in enumerate(Ks_tilde):
#             K = round(K_tilde * np.log2(N))
#             success_count = 0
#             for t in range(n_samples):
#                 A, v = createRandomGraphWithPlantedClique(N, K)
#                 truth = [i for i in range(N) if v[i] == 1]
#                 estimate, free_nrg = beliefPropagationAroundSolutionInit(
#                     A, N, K, 0.00001, v)
#                 if len([i for i in estimate if i not in truth]) + len([i for i in truth if i not in estimate]) == 0:
#                     success_count += 1
#             results[i, j] = float(success_count) / n_samples
#     return results

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
        estimate2, nrj2, info = metropolisHastingsRandomInit(
            A, N, K, n_steps, param_k=param_k)
        estimate_indices = [i for i in range(N) if estimate2[i] == 1]
        diff_not_in_truth = [i for i in estimate_indices if i not in truth]
        diff_not_in_estimate = [i for i in truth if i not in estimate_indices]
        print("Clique size in estimate:", len(estimate_indices))
        print("Count not in truth but in estimate:", len(diff_not_in_truth))
        print("Count not in estimate but in truth:", len(diff_not_in_estimate))

        print("Changes (%):", float(info["changes"]) / n_steps * 100)
        print("Cand inf (%):", float(info["candInf"]) / n_steps * 100)
        print("X inf (%):", float(info["xInf"]) / n_steps * 100)
        print("===================== END test =====================")


def testParallelTempering(N, K, param_k, betas=[], n_steps=5):
    print("===================== START test =====================")
    print("CONFIGURATION:")
    print("N", N)
    print("K", K)
    print("param_k", param_k)
    A, v = createRandomGraphWithPlantedClique(N, K)
    truth = [i for i in range(N) if v[i] == 1]
    if len(betas) == 0:
        betas = [1 - i * 0.05 for i in range(19)]
    start = datetime.now()
    estimate, monitoring_metropolis, monitoring_tempering = parallelTempering(
        A, N, K, betas, param_k, n_steps)
    stop = datetime.now()
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

    try:
        print("Elapsed time:", ceil((stop - start).seconds / 60), "min",
              (stop - start).seconds - ceil((stop - start).seconds / 60), "sec")
    except:
        pass

    try:
        tempering_switch_total_count = np.array(
            [x["switchCount"] for x in monitoring_tempering]).sum()
        tempering_switch_beta_1_count = np.array(
            [len([y for y in x["changes"] if 0 in y]) for x in monitoring_tempering]).sum()
        print("Total tempering switch:", tempering_switch_total_count)
        print("Total tempering switch for beta = 1.0:",
              tempering_switch_beta_1_count)
    except:
        pass

    try:
        print("Changes for beta = 1.0 (%):",
              monitoring_metropolis[0]["changes"] * 100)
        print("Cand inf for beta = 1.0 (%):",
              monitoring_metropolis[0]["candInf"] * 100)
        print("X inf for beta = 1.0 (%):",
              monitoring_metropolis[0]["xInf"] * 100)
    except:
        pass
    print("===================== END test =====================")


def performMetropolisOnAllReplicas(estimates, betas, A, N, K, param_k, n_steps):
    new_estimates = [np.zeros(N) for i in range(len(betas))]
    new_energies = [0 for i in range(len(betas))]
    monitoring = [{"changes": 0, "candInf": 0, "xInf": 0}
                  for _ in range(len(betas))]
    for i, beta in enumerate(betas):
        x, H_x, info = metropolisHastings(
            A, N, K, estimates[i], n_steps, beta, param_k)
        new_estimates[i] = x
        new_energies[i] = H_x
        monitoring[i] = info
    return new_estimates, new_energies, monitoring


def performSwitchConfiguration(estimates, energies, betas):
    # not sure whether change possibility for all or for pairs ?
    new_estimates = [np.copy(estimates[i]) for i in range(len(betas))]
    new_energies = [energies[i] for i in range(len(betas))]
    monitoring = {"switchCount": 0, "changes": []}
    for i in range(len(betas) - 1):
        p_switch = np.exp((betas[i] - betas[i + 1]) *
                          (new_energies[i] - new_energies[i + 1]))
        p_switch = min(1, p_switch)
        p = np.random.uniform()
        if p <= p_switch:
            monitoring["switchCount"] += 1
            monitoring["changes"].append([i, i + 1])
            new_estimates_i = new_estimates[i]
            new_estimates[i] = new_estimates[i + 1]
            new_estimates[i + 1] = new_estimates_i
            new_energies_i = new_energies[i]
            new_energies[i] = new_energies[i + 1]
            new_energies[i + 1] = new_energies_i
    return new_estimates, new_energies, monitoring


def parallelTempering(A, N, K, betas, param_k, n_steps=5):
    # monitor changes in configuration, and % of changes in metropolis by beta
    control = 0
    limit = 400
    estimates = [np.zeros(N) for i in range(len(betas))]
    energies = [0.0 for i in range(len(betas))]
    monitoring_metropolis = [
        {"changes": 0, "candInf": 0, "xInf": 0} for _ in range(len(betas))]
    monitoring_tempering = []
    size_estimate_clique = 0
    printProgressBar(0, K, prefix=f"Progress:",
                     suffix="of the clique size", length=50)
    while control < limit and size_estimate_clique < K:
        estimates, energies, new_monitoring_metropolis = performMetropolisOnAllReplicas(
            estimates, betas, A, N, K, param_k, n_steps)
        monitoring_metropolis = [{"changes": (monitoring_metropolis[i]["changes"] + float(new_monitoring_metropolis[i]["changes"]) / n_steps) / 2, "candInf": (monitoring_metropolis[i]["candInf"] + float(
            new_monitoring_metropolis[i]["candInf"]) / n_steps) / 2, "xInf": (monitoring_metropolis[i]["xInf"] + float(new_monitoring_metropolis[i]["xInf"]) / n_steps) / 2} for i in range(len(monitoring_metropolis))]
        estimates, energies, monitoring_tempering_step = performSwitchConfiguration(
            estimates, energies, betas)
        monitoring_tempering.append(monitoring_tempering_step)
        size_estimate_clique = estimates[0].sum()
        current_number_changes_temp = np.array(
            [x["switchCount"] for x in monitoring_tempering]).sum()
        printProgressBar(size_estimate_clique, K, prefix=f"Progress:",
                         suffix=f"of the clique size (step #{control}, # changes temperature: {current_number_changes_temp})", length=50)
        control += 1
    printProgressBar(K, K, prefix=f"Progress:",
                     suffix=f"of the clique size (step #{control})", length=50)
    if control >= limit:
        print("Failed to recover")
    return estimates[0], monitoring_metropolis, monitoring_tempering


if __name__ == '__main__':
    N_test, K_test = 200, 20
    param_k_test = floor(0.01 * N_test)
    # n_steps = 400
    # configs = [
    #     {
    #         "N": N,
    #         "K": K,
    #         "n_steps": n_steps,
    #         "param_k": 20
    #     },
    #     {
    #         "N": N,
    #         "K": K,
    #         "n_steps": n_steps,
    #         "param_k": 30
    #     }
    # ]
    # testMetropolis(configs)
    # testParallelTempering(N, K, 25)
    testParallelTempering(N_test, K_test, param_k_test)
    pass


"""
Remarks:
when draw candidate: do we have to check whether it is a 'good' candidate (vertex to switch has edge with each vertex of the clique) or not ? I think not but not sure
"""

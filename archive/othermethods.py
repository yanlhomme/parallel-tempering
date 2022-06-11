import numpy as np


def createRandomGraphWithPlantedClique(N, K):
    pass


def metropolisHastings(A, N, K, x_init, n_steps, log_K_over_N, log_1_minus_K_over_N, N_count_constant, param_k=0, print_progress=True):
    pass


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
    change_log_filename = "change_betas_N100_4.txt"
    if len(change_log_filename) > 0:
        with open(change_log_filename, "a") as f:
            f.write("===== New step =====\n")
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
            if len(change_log_filename) > 0:
                with open(change_log_filename, "a") as f:
                    f.write(f"beta index {i} change with {i + 1}\n")
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

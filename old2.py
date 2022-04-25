import numpy as np

from progressbar import printProgressBar

# np.random.seed(300)


def createRandomGraphWithPlantedClique(N, K):
    v_index_choices = np.random.choice(N, K, replace=False)
    v = np.array([1 if i in v_index_choices else 0 for i in range(N)])
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            elif v[i] * v[j] == 1:
                A[i, j] = 1
            else:
                p = np.random.uniform()
                if p < 0.5:
                    A[i, j] = 1
    return A, v


def betheFreeEnergy(N, z, z_ij):
    return - 1 / N * (np.log(z).sum() - (np.array([np.log(np.array([z_ij[i, j] for j in range(N) if j != i])).sum() for i in range(N)])).sum())


def beliefPropagationAroundSolutionInit(A, N, K, epsilon, v):
    psys_1 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and i in v:
                psys_1[i, j] = A[i, j]
    return beliefPropagation(A, N, K, epsilon, psys_1)


def beliefPropagationRandomInit(A, N, K, epsilon):
    psys_1 = np.abs(np.random.standard_normal((N, N)))
    np.fill_diagonal(psys_1, 0)
    return beliefPropagation(A, N, K, epsilon, psys_1)


def beliefPropagation(A, N, K, epsilon, psys_1):
    last_psys_1 = np.copy(psys_1)
    control = 0
    constant_N_K = float(K) / N * 0.5**(N - 1)
    while control < 1000 and (control == 0 or (np.absolute(last_psys_1 - psys_1) > epsilon).sum() > 0):
        printProgressBar(0, N*N, prefix=f"Progress iteration {control}:",
                         suffix='Complete', length=50)
        last_psys_1 = np.copy(psys_1)
        for i in range(N):
            for j in range(N):
                if j % 10 == 0 or j == N - 1:
                    printProgressBar(
                        i*N + j + 1, N*N, prefix=f"Progress iteration {control}:", suffix='Complete', length=50)
                if i != j:
                    A_ij_constant = (2 * A[i, j] - 1)
                    psys_1[i, j] = constant_N_K
                    for k in range(N):
                        if k != j and k != i:
                            psys_1[i, j] *= 1 + A_ij_constant * \
                                last_psys_1[k, i]
                    # psys_1[i, j] = constant_N_K * np.array(
                    #     [1 + A_ij_constant * last_psys_1[k, i] for k in range(N) if k != j and k != i]).prod()
        control += 1
        if not control < 1000:
            print("Reached control")
    print("Number of iterations:", control)
    psys_0 = (N - float(K)) / N * 0.5**N
    constant_N_K = float(K) / N * 0.5**N
    psys_results_1 = np.zeros(N)
    etha_1 = np.zeros(N)
    z = np.zeros(N)
    z_ij = np.zeros((N, N))
    for i in range(N):
        # psys_results_1[i] = constant_N_K * np.array(
        #     [1 + (2 * A[i, k] - 1) * psys_1[k, i] for k in range(N) if k != i]).prod()
        psys_results_1[i] = constant_N_K
        for k in range(N):
            if k != i:
                psys_results_1[i] *= 1 + (2 * A[i, k] - 1) * psys_1[k, i]
        z[i] = psys_0 + psys_results_1[i]
        etha_1[i] = psys_results_1[i] / z[i]
        for j in range(N):
            z_ij[i, j] = z[i] / (psys_0 * 2 + psys_1[i, j])
    bethe_free_energy = betheFreeEnergy(N, z, z_ij)
    C = np.sort(np.argpartition(etha_1, -K)[-K:])
    return C, bethe_free_energy


def figure1_stabilityOfPlantedSolution(sizes, Ks_tilde, n_samples):
    results = np.zeros((len(sizes), len(Ks_tilde)))
    for i, N in enumerate(sizes):
        for j, K_tilde in enumerate(Ks_tilde):
            K = round(K_tilde * np.log2(N))
            success_count = 0
            for t in range(n_samples):
                A, v = createRandomGraphWithPlantedClique(N, K)
                truth = [i for i in range(N) if v[i] == 1]
                estimate, free_nrg = beliefPropagationAroundSolutionInit(
                    A, N, K, 0.00001, v)
                if len([i for i in estimate if i not in truth]) + len([i for i in truth if i not in estimate]) == 0:
                    success_count += 1
            results[i, j] = float(success_count) / n_samples
    return results


if __name__ == '__main__':
    N, K = 1000, 200
    A, v = createRandomGraphWithPlantedClique(N, K)
    truth = [i for i in range(N) if v[i] == 1]
    # estimate, free_nrg = beliefPropagationRandomInit(
    #     A, N, K, 0.00001)
    estimate2, free_nrg2 = beliefPropagationAroundSolutionInit(
        A, N, K, 0.000000000001, v)
    print(truth)
    # print([i for i in estimate if i not in truth])
    diff = [i for i in estimate2 if i not in truth]
    print(diff)
    print(len(diff))
    # print(free_nrg)
    print(free_nrg2)
    pass

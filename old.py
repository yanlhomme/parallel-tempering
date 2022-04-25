import numpy as np
from sympy.stats import E, Normal
from sympy import Sum, sqrt, factorial
from sympy.abc import k

np.random.seed(200)


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


def muHat(l, d_star):
    if l == 1:
        return 1
    Z = Normal("Z", 0, 1)
    z = muHat(l - 1, d_star) + Z
    return E(p_asExpression(z, l - 1, d_star))


def p_asExpression(z, l, d_star):
    if l == 0:
        return 1
    Z = Normal("Z", 0, 1)
    mu_hat = muHat(l, d_star)
    L_hat = sqrt(
        E(Sum((mu_hat * Z)**k / factorial(k), (k, 0, d_star)).doit()**2))
    return 1 / L_hat * Sum(mu_hat**k * z**k / factorial(k), (k, 0, d_star)).doit()


def p(z, l, d_star):
    if l == 0:
        return 1
    Z = Normal("Z", 0, 1)
    mu_hat = muHat(l, d_star)
    L_hat = sqrt(
        E(Sum((mu_hat * Z)**k / factorial(k), (k, 0, d_star)).doit()**2))
    return 1 / L_hat * np.array([mu_hat**i * z**i / factorial(i) for i in range(d_star + 1)]).sum()


def powerIteration(A, t_star_star, C_tilde_N):
    u = np.ones(len(C_tilde_N))
    for t in range(t_star_star):
        u = np.dot(A, u) / np.linalg.norm(np.dot(A, u))
    return u


def zetaScore(B_N, A, rho_bar, i):
    return np.array([A[i, j] * (1 if np.abs(A[i, j]) <= rho_bar else 0) for j in B_N]).sum()


def messagePassing(A, N, K, d_star, t_star, t_star_star, rho_bar):
    W = 1 / np.sqrt(N) * A
    theta_messages = np.random.standard_normal((N, N))
    theta = np.ones(N)
    for t in range(1, t_star + 1):
        theta = np.array([np.array([W[l, i] * p(theta_messages[l, i], t - 1, d_star)
                         if l != i else 0 for l in range(N)]).sum() for i in range(N)])
        theta_messages = np.array(
            [[theta[i] - W[i, j] * p(theta_messages[j, i], t - 1, d_star) for j in range(N)] for i in range(N)])
    mu_hat_t_star = muHat(t_star, d_star)
    C_tilde_N = [i for i in range(N) if theta[i] >= mu_hat_t_star / 2]
    W_C_tilde_N = W[C_tilde_N, :][:, C_tilde_N]
    u_star_star = powerIteration(W_C_tilde_N, t_star_star, C_tilde_N)
    u_indices = np.argpartition(u_star_star, -K)[-K:]
    B_N = [i for i in range(N) if i in u_indices]
    C_hat_N = [i for i in range(N) if zetaScore(
        B_N, A, rho_bar, i) >= len(B_N) / 2]
    return C_hat_N


if __name__ == '__main__':
    N, K = 100, 60
    A, v = createRandomGraphWithPlantedClique(N, K)
    truth = [i for i in range(N) if v[i] == 1]
    estimate = messagePassing(A, N, K, 100, 100, 100, 1)
    print(truth)
    print(estimate)
    pass

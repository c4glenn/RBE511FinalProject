from typing import Tuple

import numpy as np

from swarmzbtq.param_classes import GGCQueueParams, WaitResult


def erlang_c(c: int, a: float) -> float:
    """ Compute the Erlang-C formula: P(arriving customer has to wait) in an M/M/C with offered load
    
    numerically stable recursive formulation to avoid factorial overflow on large c
    
    Args:
        c (int): Number of servers
        a (float): Offered load (Erlangs) a < c

    Returns:
        float: probability of a block [0, 1]
    """
    if a >= c:
        return 1.0 #unstable, always waiting
    
    term = 1.0
    psum = 1.0
    for k in range(1, c):
        term *= a / k
        psum += term
    
    term *= a / c
    
    numerator = term / (1.0 - a / c)
    return numerator / (psum + numerator)

def kingman_mean(params:GGCQueueParams) -> float:
    """Compute the mean waiting time in queue E[Wq] using Kingman's formula
    https://en.wikipedia.org/wiki/Kingman%27s_formula
    Args:
        params (GG1QueueParams): queue parameters

    Returns:
        float: Approximate mean waiting time in queue
    """
    assert params.c == 1, "kingman is only defined for c=1, Use whitt_ggc()"
    rho = params.rho
    mean_service = 1.0/params.mu
    return (rho / (1.0 - rho)) * ((params.ca2 + params.cs2) / 2) * mean_service

def whitt_gg1(params:GGCQueueParams) -> WaitResult:
    """Approximate the mean and standard deviation of G/G/1 waiting times using Whitt's heavy-traffic diffusion theory

    https://en.wikipedia.org/wiki/Heavy_traffic_approximation
    
    Args:
        params (GG1QueueParams): params

    Returns:
        WaitResult
    """
    
    mean_service = 1.0 / params.mu
    var_service = params.cs2 / (params.mu ** 2)
    
    Ewq = kingman_mean(params)
    if params.ca2 == 0 and params.cs2 == 0:
        phi = 0.0 #Purely deterministic 
    else:
        phi = ((2.0/3.0) * params.rho * ((params.ca2 - params.cs2)**2) / ((params.ca2+params.cs2)*(params.mu ** 2)) * (params.rho / (1.0-params.rho)) ** 2)
    
    var_wq = Ewq ** 2 + phi
    std_wq = np.sqrt(max(var_wq, 0.0))
    
    
    
    return WaitResult(
        Ewq, std_wq, params.rho, 1, erlang_c(1, params.offered_load), "Whitt heavy-traffic (G/G/1)"
    )

def whitt_ggc(params:GGCQueueParams) -> WaitResult:
    if params.c == 1: return whitt_gg1(params)
    
    a = params.offered_load
    Ec = erlang_c(params.c, a)
    mmc_Ewq = Ec / (params.c * params.mu - params.lam)
    
    f = (params.ca2 + params.cs2) / 2
    Ewq = f * mmc_Ewq
    
    if (params.ca2 == 0.0) and (params.cs2 == 0.0):
        phi = 0.0
    else: 
        phi =  ((2.0/3.0) * params.rho * ((params.ca2 - params.cs2)**2) / ((params.ca2+params.cs2)* params.c * (params.mu ** 2)) * (params.rho / (1.0-params.rho)) ** 2)

    var_Wq = Ewq ** 2 + phi
    std_wq = np.sqrt(max(var_Wq, 0.0))
    
    return WaitResult(
        mean_Wq=Ewq, std_Wq=std_wq, rho=params.rho, c=params.c, erlang_c=Ec, method=f"Whitt (1993) G/G/{params.c}"
    )

if __name__ == "__main__":
    sep = "=" * 62
 
    # ── 1. Verify G/G/1 backward compatibility ──────────────────────────────
    print(sep)
    print("G/G/1 scenarios  (c = 1)")
    print(sep)
 
    gg1_cases = [
        ("M/M/1  rho=0.80", 0.80, 1.0, 1.0, 1.0, 1),
        ("M/D/1  rho=0.80", 0.80, 1.0, 1.0, 0.0, 1),
        ("D/D/1  rho=0.80", 0.80, 1.0, 0.0, 0.0, 1),
        ("G/G/1  rho=0.95", 0.95, 1.0, 1.5, 0.5, 1),
    ]
    for label, lam, mu, ca2, cs2, c in gg1_cases:
        print(f"\n--- {label} ---")
        print(whitt_ggc(GGCQueueParams(lam=lam, mu=mu, ca2=ca2, cs2=cs2, c=c)))
 
    # ── 2. G/G/c  — show effect of adding servers ───────────────────────────
    print(f"\n{sep}")
    print("G/G/c: adding servers  (lam=3.8, mu=1.0, ca2=1.0, cs2=1.0)")
    print("       rho decreases as c increases")
    print(sep)
 
    for c in [4, 5, 6, 8, 10]:
        print(f"\n--- c = {c} ---")
        print(whitt_ggc(GGCQueueParams(lam=3.8, mu=1.0, ca2=1.0, cs2=1.0, c=c)))
 
    # ── 3. Effect of variability on a G/G/c queue ───────────────────────────
    print(f"\n{sep}")
    print("G/G/c: variability  (lam=3.8, mu=1.0, c=5, rho≈0.76)")
    print(sep)
 
    variability_cases = [
        ("D/D/5  ca2=0, cs2=0", 0.0, 0.0),
        ("M/D/5  ca2=1, cs2=0", 1.0, 0.0),
        ("M/M/5  ca2=1, cs2=1", 1.0, 1.0),
        ("G/G/5  ca2=2, cs2=1", 2.0, 1.0),
        ("G/G/5  ca2=3, cs2=3", 3.0, 3.0),
    ]
    for label, ca2, cs2 in variability_cases:
        print(f"\n--- {label} ---")
        print(whitt_ggc(GGCQueueParams(lam=3.8, mu=1.0, ca2=ca2, cs2=cs2, c=5)))
 
    # ── 4. Sanity check: M/M/c exact vs approximation ───────────────────────
    print(f"\n{sep}")
    print("Sanity check: M/M/c  (ca2=1, cs2=1 → f=1 → exact M/M/c mean)")
    print(sep)
 
    print("\nFor M/M/c, E[Wq] should equal C(c,a)/(c*mu - lam) exactly.")
    for c, lam in [(2, 1.6), (4, 3.2), (8, 6.4)]:
        res     = whitt_ggc(GGCQueueParams(lam=lam, mu=1.0, ca2=1.0, cs2=1.0, c=c))
        exact   = res.erlang_c / (c * 1.0 - lam)
        print(
            f"  c={c}, lam={lam}  "
            f"approx E[Wq]={res.mean_Wq:.6f}  "
            f"exact E[Wq]={exact:.6f}  "
            f"match={np.isclose(res.mean_Wq, exact, atol=1e-9)}"
        )
 

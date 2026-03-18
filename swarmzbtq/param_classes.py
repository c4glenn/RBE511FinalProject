from dataclasses import dataclass

@dataclass
class GGCQueueParams:
    """Parameters for a G/G/c Queue"""
    lam: float
    """Arrival Rate (customers per unit time)"""
    mu: float
    """Service Rate (cusomers per unit time)"""
    ca2: float
    """squared coeff of variation of interarrival times ca2 = Var(A) / E[A]^2 - for Poisson ca2 = 1"""
    cs2: float
    """squared coeff of variation of service times cs2 = Var(S) / E[S]^2 - for deterministic cs2 = 0"""
    c: int = 1
    """Number of parallel servers (defualt 1 -> G/G/1)"""
    def __post_init__(self):
        assert self.lam > 0, "arrival rate must be postive"
        assert self.mu > 0, "service rate must be postive"
        assert self.ca2 >= 0, "squared coeff has to be positive"
        assert self.cs2 >= 0, "squared coeff has to be positive"
        
        if self.rho >= 1:
            raise ValueError(f"server utilization ({self.rho=:.4f}) >= 1", "Queue is unstable; waiting times are infinite")
        
    @property
    def rho(self) -> float:
        """Traffic intensity aka Per-Server utilization (p->1 = High traffic conditions)"""
        return self.lam / (self.c * self.mu)
    
    @property
    def offered_load(self) -> float:
        """Total offered load a = lam / mu in erlangs"""
        return self.lam / self.mu
        
@dataclass
class WaitResult:
    """Results from a waiting time approximation"""
    mean_Wq: float        
    """E[Wq]  — mean time waiting in queue"""
    std_Wq: float         
    """Std dev of Wq"""
    rho: float            
    """Per-Server utilisation"""
    c: int
    """number of servers"""
    erlang_c: float
    """P(wait) - Erlang-C blocking probability"""
    method: str           
    """Name of the approximation used"""

from .fps import fps_smc, fps_spf
from .latent_latino import latent_latino
from .mala_sal import mala_sal
from .nsps import nsps
from .oracle_langevin import oracle_langevin
from .sal import score_annealed_langevin

SOLVERS = {
    "FPS-SMC": fps_smc,
    "FPS-SPF": fps_spf,
    "Latent LATINO": latent_latino,
    "MALA-SAL": mala_sal,
    "NSPS": nsps,
    "Oracle Langevin": oracle_langevin,
    "SAL": score_annealed_langevin,
}

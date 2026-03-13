from .fps import fps_smc, fps_spf
from .latent_latino import latent_latino
from .oracle_langevin import oracle_langevin

SOLVERS = {
    "FPS-SMC": fps_smc,
    "FPS-SPF": fps_spf,
    "Latent LATINO": latent_latino,
    "Oracle Langevin": oracle_langevin,
}

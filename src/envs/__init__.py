from src.envs.cycle import SquareEnvironment
from src.envs.isosceles import IsoscelesEnvironment
from src.envs.orbital import OrbitalEnvironment
from src.envs.sphere import SphereEnvironment

ENVS = {"square": SquareEnvironment, "isosceles": IsoscelesEnvironment, "sphere": SphereEnvironment, "orbit": OrbitalEnvironment}


def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)
    return env

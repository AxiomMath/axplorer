from src.envs.cycle import SquareEnvironment
from src.envs.isosceles import IsoscelesEnvironment
from src.envs.sphere import SphereEnvironment
from src.envs.ramsey import RamseyEnvironment

ENVS = {
    "square": SquareEnvironment,
    "isosceles": IsoscelesEnvironment,
    "sphere": SphereEnvironment,
    "ramsey": RamseyEnvironment,
}


def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)
    return env

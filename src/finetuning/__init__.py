import warnings

# These are needed to silence some warnings from pydantic before the silosci import.
warnings.filterwarnings(
    "ignore", message='Field "model_server_url" has conflict with protected namespace "model_".', category=UserWarning
)
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:", category=UserWarning)

from finetuning import cli, config, model  # noqa: E402
from finetuning.model import model_to_peft_model  # noqa: E402

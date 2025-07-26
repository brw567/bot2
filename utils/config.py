from typing import Any, Dict
from db_utils import get_param
import config as default_config


def load_config_from_db() -> Dict[str, Any]:
    """Load configuration values from the database with defaults."""
    cfg: Dict[str, Any] = {}
    for key, val in default_config.DEFAULT_PARAMS.items():
        raw = get_param(key, val)
        try:
            cfg[key] = type(val)(raw)
        except Exception:
            cfg[key] = val
    return cfg

from db_utils import get_param
import config as default_cfg

# Default analytics-related parameters
DEFAULTS = {
    'max_active_pairs': default_cfg.DEFAULT_PARAMS.get('auto_pair_limit', 5),
    'swap_multiplier': default_cfg.DEFAULT_PARAMS.get('swap_pair_multiplier', 10),
    'grok_interval': default_cfg.DEFAULT_PARAMS.get('grok_interval', 4 * 60 * 60),
    'dune_interval': default_cfg.DEFAULT_PARAMS.get('dune_interval', 600),
    'analytics_interval': default_cfg.DEFAULT_PARAMS.get('analytics_interval', 60),
    'swap_threshold': default_cfg.DEFAULT_PARAMS.get('swap_threshold', 1.5),
    'cooldown': default_cfg.DEFAULT_PARAMS.get('cooldown', 45 * 60),
    'forecast_period': default_cfg.DEFAULT_PARAMS.get('forecast_period', 4 * 60 * 60),
    'history_period': default_cfg.DEFAULT_PARAMS.get('history_period', 24 * 60 * 60),
}


def _cast(value, default):
    """Cast DB strings to the type of the default value."""
    if isinstance(default, bool):
        return str(value).lower() == 'true'
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def load_config_from_db() -> dict:
    """Load analytics configuration from DB with config.py fallbacks."""
    cfg = {}
    for key, default in DEFAULTS.items():
        val = get_param(key, default)
        cfg[key] = _cast(val, default)
    return cfg

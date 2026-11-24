import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / 'construct_config.json'
_config = None
_config_path = DEFAULT_CONFIG_PATH


def configure_construct(config_path=None):
    """Load construction settings from a JSON file, resolving its path from the project root."""
    global _config, _config_path
    path = resolve_project_path(config_path or DEFAULT_CONFIG_PATH)
    with path.open(encoding='utf-8-sig') as stream:
        settings = json.load(stream)
    if not isinstance(settings, dict):
        raise ValueError(f'Construction configuration must be a JSON object: {path}')
    _config = settings
    _config_path = path
    return settings


def configure_from_cli(argv=None):
    """Select the construction config before importing modules that initialize model clients."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    arguments, _ = parser.parse_known_args(argv)
    return configure_construct(arguments.config)


def get_setting(key):
    """Read a required setting using a dot-separated key such as 'fusion.dbscan.eps'."""
    if _config is None:
        configure_construct()
    value = _config
    try:
        for part in key.split('.'):
            value = value[part]
    except (KeyError, TypeError) as error:
        raise KeyError(f'Missing construction setting {key!r} in {_config_path}') from error
    return value


def get_neo4j_url():
    """Build a Neomodel URL from separate credentials, accepting legacy authenticated URLs."""
    settings = get_setting('neo4j')
    parsed = urlsplit(settings['url'])
    if not parsed.scheme or not parsed.hostname:
        raise ValueError('neo4j.url must include a protocol and server address.')
    username = settings.get('username', unquote(parsed.username or ''))
    password = settings.get('password', unquote(parsed.password or ''))
    if not isinstance(username, str) or not isinstance(password, str):
        raise ValueError('neo4j.username and neo4j.password must be strings.')
    server = parsed.netloc.rsplit('@', 1)[-1]
    credentials = f'{quote(username, safe="")}:{quote(password, safe="")}@' if username or password else ''
    return urlunsplit(parsed._replace(netloc=credentials + server))


def get_config_path():
    """Return the active construction configuration path."""
    return _config_path


def resolve_project_path(value):
    """Resolve a configured or explicitly supplied file path relative to the project root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def get_configured_path(key, create_parent=False):
    """Resolve a configured file path and optionally create its output parent directory."""
    path = resolve_project_path(get_setting(key))
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_chat_model_options(name):
    """Return model options, allowing the configured environment variable to supply the API key."""
    options = dict(get_setting(f'llm.{name}'))
    environment_variable = options.pop('api_key_env', None)
    environment_key = os.environ.get(environment_variable) if environment_variable else None
    if environment_key:
        options['api_key'] = environment_key
    return options

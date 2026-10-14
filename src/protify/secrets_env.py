"""Programmatic access to the repository secrets file.

Synthyra repositories keep credentials in a gitignored `.secrets.env` at the repository
root. Loading it into the process environment lets every downstream consumer keep reading
`os.environ` without knowing whether a value came from the file or an exported variable.

The loader is `foundry.secrets_env`, vendored at the repository root, whose docstring states how
the file is found and read. `load_secrets_env(start)` searches upward from `start`. Callers
receive variable names only. Never log or print a loaded value.
"""

from foundry.secrets_env import SECRETS_FILENAME, find_secrets_file, load_secrets_env, parse_secrets_file


__all__ = ["SECRETS_FILENAME", "find_secrets_file", "load_secrets_env", "parse_secrets_file"]

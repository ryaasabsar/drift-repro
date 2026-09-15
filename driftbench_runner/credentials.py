"""Load the local Hugging Face credential file without executing shell code."""
import os
from pathlib import Path
import shlex
from .common import ROOT

CREDENTIALS = ROOT / '.env'


def ensure_credentials_file(path=CREDENTIALS):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path
    with os.fdopen(descriptor, 'w') as handle:
        handle.write('HF_TOKEN=\n')
    return path


def load_credentials(path=CREDENTIALS):
    """Read data, never source shell code or include secret values in errors."""
    path = ensure_credentials_file(path)
    try:
        data = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:].lstrip()
            key, separator, value = line.partition('=')
            key = key.strip()
            if not separator or key != 'HF_TOKEN' or key in data:
                raise ValueError
            values = shlex.split(value, comments=True, posix=True)
            if len(values) > 1:
                raise ValueError
            data[key] = values[0] if values else ''
    except (OSError, ValueError):
        raise ValueError('Cannot read credentials: .env must contain an HF_TOKEN assignment') from None
    token = data.get('HF_TOKEN', '').strip()
    if token:
        os.environ['HF_TOKEN'] = token


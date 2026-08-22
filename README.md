# Roast-My-Outfit

Project scaffolding. Nothing is implemented yet beyond package installation, path resolution and
dataset staging.

## Requirements

Python >= 3.11.

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux, `source .venv/bin/activate` instead.

## Tests

```powershell
pytest -q
```

## Dataset

DeepFashion-MultiModal sits behind a Google Drive consent screen, so the download is manual and
only has to be done once. Take the labels archive and the captions JSON (~11.5 MB) from
<https://github.com/yumingj/DeepFashion-MultiModal>, unpack them anywhere, then stage them:

```powershell
python scripts/download_data.py --from C:\datasets\deepfashion-mm
```

Files are copied into `data/raw/` (gitignored) and a summary of what is staged is written to the
log.

```powershell
python scripts/download_data.py --verify                       # report what is staged
python scripts/download_data.py --images --limit 200 --from .  # small image sample
python scripts/download_data.py --all --from .                 # everything, ~5.5 GB
```

Exit codes: `0` success, `2` no source directory, `3` a requested group matched no files.

The dataset is non-commercial research use only and may not be redistributed, so `data/raw/` is
never committed.

## Layout

```
src/rmo/     package
scripts/     CLI wrappers
tests/       pytest
data/        gitignored
```

| Environment variable | Effect |
|---|---|
| `RMO_ROOT` | Repository root, when `pyproject.toml` cannot be found by walking up |
| `RMO_DATA_ROOT` | Data tree, e.g. a read-only mount |
| `RMO_SOURCE_DIR` | Default for `download_data.py --from` |


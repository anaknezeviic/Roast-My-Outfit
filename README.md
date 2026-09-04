# Roast-My-Outfit

An outfit photograph goes in, a structured description comes out, then a style-compatibility score,
then a roast. The three stages are pluggable and meet at the records described in
[docs/CONTRACTS.md](docs/CONTRACTS.md). Only fixture-replay stage implementations exist so far.

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

## Pipeline

```powershell
python -m rmo.pipeline --image data/fixtures/images/fixture_000.png
```

One `RoastOutput` is written to stdout as JSON. Any stage left unregistered falls back to its
fixture-replay model. Add `--json` to get the description and the score alongside the roast.

## Gemini roast integration

The rule-based roaster can also be replaced with a hosted Gemini roast generator. Copy the
committed environment template, add your Google AI Studio key, then select `gemini_roaster`:

```powershell
Copy-Item .env.example .env
# Edit .env and set GEMINI_API_KEY. Never commit .env.
python -m rmo.pipeline --image data/fixtures/images/fixture_002.png --roaster gemini_roaster --json
```

`GeminiRoaster` uses `gemini-3.6-flash` by default. Set `RMO_GEMINI_MODEL` in `.env` to override
the model. If the API fails, returns invalid structured output, or fails the local safety checks,
the pipeline falls back to `RuleBasedRoaster`.

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

## Outfit table and splits

Once the labels, captions and parsing masks are staged, build the canonical table and the
group-aware splits:

```powershell
python scripts/build_dataset.py                  # data/processed/outfits.parquet
python scripts/build_dataset.py --splits         # also train/val/test and their manifest
```

Splits are keyed by product, not by image, so every shot of one garment lands on the same side.
They are written once and committed; `--splits` refuses to overwrite them, because every reported
metric is keyed to the files that exist. Exit codes: `0` success, `2` unusable input, `3` splits
already exist.

## Layout

```
src/rmo/     package
scripts/     CLI wrappers
tests/       pytest
docs/        record and registry contracts
data/        fixtures and splits committed, the corpus gitignored
```

| Environment variable | Effect |
|---|---|
| `RMO_ROOT` | Repository root, when `pyproject.toml` cannot be found by walking up |
| `RMO_DATA_ROOT` | Data tree, e.g. a read-only mount |
| `RMO_SOURCE_DIR` | Default for `download_data.py --from` |


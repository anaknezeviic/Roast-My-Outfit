# Roast-My-Outfit

An outfit photograph goes in, a structured description comes out, then a style-compatibility score,
then a roast. The three stages are independent, swappable implementations that meet at versioned
Pydantic records described in [docs/CONTRACTS.md](docs/CONTRACTS.md).

| Stage | Input | Output |
|---|---|---|
| `PerceptionModel` | image | `OutfitDescription` |
| `ScoringModel` | `OutfitDescription` | `OutfitScore` |
| `RoastGenerator` | `OutfitDescription` and `OutfitScore` | `RoastOutput` |

The roast generator reads both earlier records, so every roast is grounded in the garments and
issues the earlier stages produced.

Every stage ships a fixture-replay implementation and a working local implementation, and perception
and roasting additionally offer a learned or hosted implementation. Stages are selected by name at
the command line, so the pipeline runs end to end on a clean checkout with no dataset, no checkpoint
and no API key.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Stages and models](#stages-and-models)
- [Gemini roast integration](#gemini-roast-integration)
- [Records and schema](#records-and-schema)
- [Scoring model](#scoring-model)
- [Safety](#safety)
- [Dataset](#dataset)
- [Outfit table and splits](#outfit-table-and-splits)
- [Training](#training)
- [Evaluation](#evaluation)
- [Fixtures](#fixtures)
- [Configuration](#configuration)
- [Environment variables](#environment-variables)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Library use](#library-use)
- [License](#license)

## Requirements

Python 3.11 or newer. The base install runs on CPU and needs no GPU, no dataset and no network
access at runtime.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

`requirements.txt` installs the package in editable mode along with the analysis and notebook tools.
For a leaner install, use the extras declared in `pyproject.toml`:

| Install | Command | Adds |
|---|---|---|
| Base | `pip install -e .` | numpy, pandas, pillow, pyarrow, pydantic, pyyaml, requests, scikit-learn, google-genai, python-dotenv |
| Dev | `pip install -e ".[dev]"` | pytest |
| VLM | `pip install -e ".[vlm]"` | torch, torchvision, transformers |
| CNN | `pip install -e ".[cnn]"` | everything in `vlm`, plus timm |

The `vlm` and `cnn` extras are required only by the `smolvlm` and `cnn_multihead_v1` perception
models. Tests that need them skip themselves when the packages are absent.

## Quick start

```powershell
python -m rmo.pipeline --image data/fixtures/images/fixture_000.png
```

One `RoastOutput` is written to stdout as a single line of JSON. Logs go to stderr, so stdout carries
only the document and stays safe to pipe.

```powershell
python -m rmo.pipeline --image data/fixtures/images/fixture_002.png --scorer rule_scorer_v1 --roaster rule_roaster --json
```

| Flag | Description |
|---|---|
| `--image PATH` | Photograph to run through the stages. Required. |
| `--json` | Emit the description and the score alongside the roast. |
| `--perception NAME` | Registered perception model name. Defaults to `dummy_perception`. |
| `--scorer NAME` | Registered scoring model name. Defaults to `dummy_scorer`. |
| `--roaster NAME` | Registered roast generator name. Defaults to `dummy_roaster`. |
| `--verbose` | Log at DEBUG, which includes full hosted request and response bodies. |

Exit codes: `0` on success, `2` for a usage error such as a missing `--image`.

With `--json` the output document is an object with three keys, `description`, `score` and `roast`,
each holding the serialised record for that stage.

## Stages and models

Each stage package registers its implementations with `rmo.pipeline.register` as an import side
effect, so adding a model never requires editing `pipeline.py`. `rmo.pipeline.registered_names()`
returns every available name.

| Stage | Name | Implementation | Requirements |
|---|---|---|---|
| Perception | `dummy_perception` | `DummyPerception`, replays the committed fixture descriptions | base install |
| Perception | `cnn_multihead_v1` | `CNNPerception`, nine-head frozen-backbone attribute CNN | `cnn` extra and a trained checkpoint |
| Perception | `smolvlm` | `SmolVLMPerception`, local zero-shot SmolVLM | `vlm` extra, downloads weights on first use |
| Scoring | `dummy_scorer` | `DummyScorer`, replays the committed fixture scores | base install |
| Scoring | `rule_scorer_v1` | `RuleScorer`, hand-authored colour, formality, season and proportion rules | base install |
| Roast | `dummy_roaster` | `DummyRoaster`, replays the committed fixture roasts | base install |
| Roast | `rule_roaster` | `RuleBasedRoaster`, deterministic templates grounded in the scored issues | base install |
| Roast | `gemini_roaster` | `GeminiRoaster`, hosted generation with a local fallback | `GEMINI_API_KEY` |

The dummy models replay records keyed by `image_id`, so they answer only for the committed fixture
images. Passing an unregistered name raises a `ValueError` listing the valid names.

`LearnedScorer` under the name `learned_scorer_v1` is available as a class. It stays out of the
registry because it needs a fitted bundle at construction time, so it is built directly from Python
or through the training entry point described under [Training](#training).

### Perception implementations

`SmolVLMPerception` prompts a local vision language model with the instruction block in
`configs/perception.yaml`, then feeds the free text through `rmo.perception.postprocess`, which
normalises synonyms into the schema enums and never raises. Weights load lazily on first call, and
the instance reports its checkpoint id as `source_model`.

`CNNPerception` wraps a nine-head classifier over a frozen timm backbone. The heads cover
`upper_fabric`, `lower_fabric`, `outer_fabric`, `upper_pattern`, `lower_pattern`, `outer_pattern`,
`sleeve_length`, `lower_length` and `neckline`. Predictions are projected onto a garment skeleton
derived from the 24-class parsing mask, so slots the mask never saw are left out. Checkpoints carry
a format tag and a version, and loading rejects anything that fails to match. The checkpoint path
comes from `configs/perception_cnn.yaml`.

Both models share the mask and palette mechanics in `rmo.perception.enrichment`, so the garment
slots, area fractions and CIELAB colours they report are computed the same way.

## Gemini roast integration

Copy the committed environment template, add a Google AI Studio key, then select `gemini_roaster`:

```powershell
Copy-Item .env.example .env
# Edit .env and set GEMINI_API_KEY. Never commit .env.
python -m rmo.pipeline --image data/fixtures/images/fixture_002.png --roaster gemini_roaster --json
```

The default model is `gemini-3.6-flash`. Set `RMO_GEMINI_MODEL` to override it. The persona and the
per-request prompt template live in `configs/roast.yaml`. The persona restricts the model to the
outfit itself and forbids any comment on the wearer, and the template grounds every request in the
garments and issues supplied by the earlier stages.

`.env` is loaded without overriding real environment variables, so the process environment always
wins. If the API call fails, returns structured output that fails validation, or produces text that
the local safety classifier rejects, the roaster falls back to `RuleBasedRoaster` and keeps the
safety flags that caused the rejection. Without a key the roaster still constructs and answers
through the fallback, which keeps the demo runnable on a clean checkout.

## Records and schema

All boundary records live in [src/rmo/schemas.py](src/rmo/schemas.py) and carry
`schema_version = "1.0.0"`. Every model forbids extra fields, validates on assignment and strips
surrounding whitespace from strings.

| Record | Produced by | Key fields |
|---|---|---|
| `OutfitDescription` | `PerceptionModel.predict` | `image_id`, `image_path`, `garments`, `caption`, `provenance`, `source_model` |
| `OutfitScore` | `ScoringModel.score` | `image_id`, `overall`, `subscores`, `issues`, `provenance`, `source_model` |
| `RoastOutput` | `RoastGenerator.generate` | `image_id`, `roast`, `suggestions`, `tone`, `grounded_garments`, `safety_flags`, `provenance`, `source_model` |

`Garment`, `SubScores` and `Issue` appear only nested inside those three.

Enumerations: `Provenance` (`gt`, `predicted`, `fixture`), `GarmentSlot` (15 members),
`Pattern` (8), `Fabric` (8), `SleeveLength` (6), `LowerLength` (5), `Neckline` (7), `ColorName` (19),
`IssueSeverity` (`info`, `minor`, `major`), `IssueCode` (12) and `Tone` (`gentle`, `playful`,
`savage`, `compliment`). All of them subclass `str`, so interpolate `.value` when building text.

### Invariants

- `image_id` is identical across the description, the score and the roast for one run.
- A blank `Garment.ref` is auto-numbered per slot, giving `upper_0`, `upper_1` and so on. Refs are
  unique within a description and are the only stable handle a score or a roast may cite, because
  garment order carries no guarantee.
- Every `Issue.garment_refs` entry and every `RoastOutput.grounded_garments` entry must be a ref of
  the description being scored.
- Provenance flows forward. `OutfitScore` and `RoastOutput` copy the provenance of their input, so a
  fixture description can never yield a record marked `predicted`.

### Schema evolution

The policy is additive. Adding a defaulted field or a new enum member is safe. Removing, renaming or
tightening a field breaks every record ever written and requires agreement first.
[tests/test_schema_compat.py](tests/test_schema_compat.py) replays the frozen corpus
`data/fixtures/golden_v1.0.0.jsonl` through the current `OutfitDescription`. Extra keys are ignored,
so additions pass and removals fail. Regenerating the golden corpus only hides breakage, so
`make_fixtures.py` preserves an existing golden file.

## Scoring model

`rule_scorer_v1` produces four subscores in the range 0 to 100 and combines them into `overall`
using the weights in `configs/scoring.yaml`.

| Subscore | What it measures |
|---|---|
| `color_harmony` | Hue relations between garments in CIELAB, lightness contrast, hue family count, pattern load |
| `formality_consistency` | Spread across a formality table covering roughly forty garment categories |
| `seasonality` | Fabric warmth against sleeve and length exposure |
| `proportion` | Accessory count, dominant garment area, upper to lower area ratio, footwear presence |

Colours come from k-means palette extraction over the parsing mask, converted to CIELAB. Hue
relations are classified as complementary, analogous, a triadic candidate, or none, using the angular
tolerances in the config. Each rule that fires deducts a configured penalty and emits an `Issue`
carrying a code, a severity, a short message and the garment refs it applies to. Outfits with too
little measurable signal fall back to the configured `unscorable` value.

`rmo.scoring.features` builds a fixed-width feature vector over the slots and blocks named in the
config, along with a hashed contract so a stored vector can be checked against the spec that
produced it. `verify_contract` rejects any bundle whose stored contract disagrees with the current
spec.

### Learned scorer

`learned_scorer_v1` keeps the rule subscores and issues and replaces `overall` with a fitted
probability. The estimator is a standardised logistic regression over the same feature vector, and
it is persisted as a portable JSON bundle at `models/scoring/logreg.json` holding the kind, the
feature names, the weights, the intercept, the standardisation statistics, the feature contract, the
seed and the fitted row count. Loading verifies the bundle version and the contract, so a bundle
fitted under a stale feature spec is refused.

### Pair construction

Supervision comes from split-local pairs. For each observed outfit, `rmo.scoring.pairs` synthesises
negatives by swapping garments in from donor outfits drawn from a different product group, producing
one hard negative and one easy negative per recipient. Observed combinations are labelled 1 and
synthesised ones 0. Donors are only accepted when the substitution actually changes the outfit, and
`attrition_report` records how many candidates were lost at each step so the pair counts can be
stated honestly.

### Missingness guard

A scorer could reach a high score by learning which fields happen to be absent.
`rmo.scoring.missingness` fits a logistic regression on the presence and measurement indicator
columns alone and measures its validation AUC. The gate passes below `MAX_MISSINGNESS_AUC = 0.6`,
and an AUC under `0.4` is reported as an inverse warning. Fitting the learned scorer is refused
while the guard fails, and the guard evidence is written to
`results/metrics/scoring_missingness_guard.json` regardless of the outcome.

## Safety

`rmo.roast.safety.flag_text` is a regex classifier over roast text covering eight categories:
`body`, `age`, `race`, `gender`, `attractiveness`, `disability`, `profanity` and `implicature`.
Matches are returned as sorted `safety:<category>` strings and land in `RoastOutput.safety_flags`.

`data/fixtures/safety_probes.jsonl` holds 60 labelled probes: 40 that must flag across the eight
categories, and 20 garment critiques that must stay clean. The clean probes guard against a
classifier that suppresses legitimate criticism of clothing.

## Dataset

The project trains and evaluates on DeepFashion-MultiModal. The archive sits behind a Google Drive
consent screen, so the download is manual and only has to be done once. Take the labels archive and
the captions JSON from <https://github.com/yumingj/DeepFashion-MultiModal>, unpack them anywhere,
then stage them:

```powershell
python scripts/download_data.py --from C:\datasets\deepfashion-mm
```

Files are copied into `data/raw/`, which is gitignored, and a summary of what is staged is written to
the log. Copies are atomic and skip files already present at the same size, so the command is safe to
re-run.

```powershell
python scripts/download_data.py --verify                       # report what is already staged
python scripts/download_data.py --images --limit 200 --from .  # small image sample
python scripts/download_data.py --parsing --from .             # segmentation masks
python scripts/download_data.py --all --from .                 # everything
```

| Flag | Description |
|---|---|
| `--from DIR` | Directory holding the unpacked download. Defaults to `$RMO_SOURCE_DIR`. |
| `--labels-only` | Stage annotations and captions only. This is the default. |
| `--parsing` | Also stage the 24-class segmentation masks. |
| `--images` | Also stage the source photographs. |
| `--limit N` | Cap how many image files are staged. |
| `--all` | Stage every group. Implies `--parsing --images`. |
| `--verify` | Skip staging and report what is already in `data/raw`. |
| `--dry-run` | Report what would be staged without copying anything. |

Asset groups and approximate sizes:

| Group | Destination | Size |
|---|---|---|
| `labels` | `data/raw/labels/` | 575 KB across 3 files |
| `captions` | `data/raw/` | 11 MB |
| `parsing` | `data/raw/parsing/` | 90 MB across 12,701 masks |
| `images` | `data/raw/images/` | 5.4 GB across 44,096 photographs |

DensePose and keypoint assets are skipped. `--labels-only` cannot be combined with `--parsing`,
`--images` or `--all`.

Exit codes: `0` success, `2` no source directory, `3` a requested group matched no files or an
expected label file is missing.

The dataset is licensed for non-commercial research use and may not be redistributed, so `data/raw/`
is never committed.

## Outfit table and splits

Once the labels, captions and parsing masks are staged, build the canonical table and the
group-aware splits:

```powershell
python scripts/build_dataset.py                  # data/processed/outfits.parquet
python scripts/build_dataset.py --splits         # also train/val/test and their manifest
python scripts/build_dataset.py --limit 500      # smoke run over the first 500 rows
```

The builder decodes the integer-coded shape, fabric and pattern label files, merges them one to one,
recovers `garment_id`, `gender` and category from the filename, joins the captions, and records
whether a parsing mask and a full-body shot exist for each image. The result is written atomically to
`data/processed/outfits.parquet`.

Splits are keyed by product, so every shot of one garment lands on the same
side. They are produced with a stratified grouped 7-fold partition under a fixed seed, taking one
fold as test, one as validation and the rest as train. Split files and their manifest are committed,
and `--splits` refuses to overwrite an existing set, because every reported metric is keyed to the
files on disk. `--force` overrides that and invalidates every previously reported number.

Exit codes: `0` success, `2` unusable input, `3` splits already exist.

## Training

### Perception CNN

```powershell
python scripts/train_cnn.py
python scripts/train_cnn.py --limit 500 --device cpu    # smoke run
```

| Flag | Description |
|---|---|
| `--config PATH` | Training config. Defaults to `configs/perception_cnn.yaml`. |
| `--table PATH` | Outfit table. Defaults to `data/processed/outfits.parquet`. |
| `--checkpoint PATH` | Checkpoint destination. Defaults to the path in the config. |
| `--device DEVICE` | Torch device. |
| `--limit N` | Cap how many images each split contributes, for a smoke run. |
| `--metrics-out PATH` | Metric destination. |

Training fits the nine attribute heads over a frozen backbone, so backbone features are extracted
once and reused. Defaults come from the `training` block of `configs/perception_cnn.yaml`:
`resnet50`, 30 epochs, batch size 256, extraction batch size 32, learning rate 0.01, weight decay
0.0001, 4 workers and a fixed seed. Loss is masked cross entropy, so a head contributes only for
samples that carry a label for it, and class weighting compensates for the rare classes. The
checkpoint kept is the epoch with the best mean macro-F1 on the validation split, and heads that saw
no usable training signal are recorded as untrained. Interrupted runs resume from a sidecar file
that is removed on success. Requires the `cnn` extra.

### Learned scorer

```powershell
python scripts/train_scoring.py --build-pairs
python scripts/train_scoring.py --guard
python scripts/train_scoring.py --fit
```

| Flag | Description |
|---|---|
| `--build-pairs` | Build the split-local pairs and log the attrition report. |
| `--guard` | Also run the missingness guard and write its evidence. |
| `--fit` | Also fit the scorer and write its bundle, provided the guard passes. |
| `--seed N` | Seed for pair construction and fitting. Defaults to `20260101`. |
| `--limit N` | Cap how many outfits each split contributes. |
| `--table PATH` | Outfit table. Defaults to `data/processed/outfits.parquet`. |
| `--bundle-out PATH` | Bundle destination. Defaults to `models/scoring/logreg.json`. |
| `--metrics-out PATH` | Guard evidence destination. |
| `--coefficients-out PATH` | Standardised coefficient export for analysis. |

At least one of `--build-pairs`, `--guard` or `--fit` is required. Pairs are built on `train` and
`val`, the guard is fitted on `train` and evaluated on `val`, and the scorer is fitted on `train`
only. A failing guard returns exit code `1` and blocks the fit.

## Evaluation

Every evaluation writes a metric record carrying a run stamp with the git commit, the schema
version, an eight-character hash of the run config, the seed, a chance baseline where one applies,
and the split inputs with their checksums. Any reported number is therefore traceable to the exact
files that produced it. Records land under `results/metrics/`.

### Perception

```powershell
python -m rmo.eval.perception_eval --perception smolvlm
python -m rmo.eval.perception_eval --perception cnn_multihead_v1 --predictions-out data/processed/predictions/cnn.jsonl
```

| Flag | Description |
|---|---|
| `--perception NAME` | Registered perception model to evaluate. Defaults to `smolvlm`. |
| `--model-id ID` | Override the checkpoint id for the VLM model. |
| `--device DEVICE` | Torch device for the evaluated model. |
| `--metrics-out PATH` | Metric destination. Defaults to `results/metrics/perception_<model>.json`. |
| `--predictions-out PATH` | Optional JSONL export of the raw predictions. |
| `--cache PATH` | Incremental cache location. Defaults to a sidecar under `data/processed/predictions/`. |
| `--fresh` | Delete the cache first and evaluate every image again. |

Evaluation runs on the held-out test split and reports per-field accuracy, macro-F1, comparison
counts and confusion tables, alongside schema validity and the count of valid generations. Fields
with no valid comparisons are reported as undefined. Descriptions that came back as postprocessing
fallbacks are excluded from field metrics and counted separately, so a model that fails to parse
cannot look accurate.

Slow generative runs are incremental. Each prediction is appended to the cache as it completes, a
torn final line is discarded on reload, and a stopped run resumes from what the cache already holds.
The cache is removed once the run finishes.

### Scoring

```powershell
python -m rmo.eval.scoring_eval --scorer rule_scorer_v1 --split val
```

| Flag | Description |
|---|---|
| `--scorer NAME` | Registered scoring model to evaluate. Defaults to `rule_scorer_v1`. |
| `--split NAME` | Split to evaluate. Defaults to `val`. |
| `--seed N` | Seed for pair construction. Defaults to `20260101`. |
| `--limit N` | Cap how many outfits contribute. |
| `--table PATH` | Outfit table. |
| `--metrics-out PATH` | Metric destination. |

Scoring is measured as discrimination between observed outfits and the synthesised negatives built
for the same split. The report carries AUC and accuracy against the hard and the easy negatives
separately as well as pooled, a paired ranking accuracy, the mean score of each group, and the
counts behind every figure. Chance is 0.5, and it is recorded in the metric record as the baseline.

### Roast

```powershell
python -m rmo.eval.roast_eval --roaster rule_roaster --split val
```

| Flag | Description |
|---|---|
| `--roaster NAME` | Registered roast generator to evaluate. Defaults to `rule_roaster`. |
| `--scorer NAME` | Scorer feeding the generator. Defaults to `rule_scorer_v1`. |
| `--split NAME` | Split to evaluate. Defaults to `val`. |
| `--limit N` | Cap how many outfits contribute. |
| `--table PATH` | Outfit table. |
| `--metrics-out PATH` | Metric destination. |

Two things are measured. Grounding reports the share of roasts that cite at least one valid garment
ref, the share citing a ref the description never contained, the share citing nothing, and the mean
number of grounded refs. Safety runs the classifier over `data/fixtures/safety_probes.jsonl` and
reports recall on the probes that must flag, the false positive rate on the garment critiques that
must stay clean, and the id of every miss and every false positive.

### Prediction export

```powershell
python -m rmo.eval.cnn_predictions --perception cnn_multihead_v1 --splits train val test
```

| Flag | Description |
|---|---|
| `--perception NAME` | Registered perception model. Defaults to `cnn_multihead_v1`. |
| `--splits NAME...` | Splits to export. Defaults to all three. |
| `--out-dir PATH` | Destination directory. Defaults to `data/processed/predictions/`. |
| `--limit N` | Cap how many images each split contributes. |

Predictions are written as JSONL beside a sidecar manifest recording the record count and a content
hash, so a stored prediction file can be checked before it is trusted.

## Fixtures

`data/fixtures/` holds a committed corpus of 28 outfits: 17 ordinary cases, 3 adversarial cases and 8
degenerate cases. Each has a rendered 128x256 PNG, an `OutfitDescription`, an `OutfitScore` and a
`RoastOutput`, all marked with `provenance: fixture`. This corpus backs the dummy stages, the
end-to-end contract tests and the schema compatibility guard, and it is the reason the pipeline runs
without the dataset.

```powershell
python scripts/make_fixtures.py
```

Regeneration is deterministic under a fixed seed. Images are rendered as colour bands sized by
garment weight, and each garment's CIELAB colour is measured from its own band. The golden corpus is
preserved if it already exists.

## Configuration

Behaviour is driven by YAML under `configs/`, loaded through cached readers in
[src/rmo/config.py](src/rmo/config.py). The loaders return shared read-only mappings, and typed
structures are built from them by the consuming module.

| File | Contents |
|---|---|
| `perception.yaml` | VLM prompt, generation settings, free-text to enum synonym tables, slot to mask-index map |
| `perception_cnn.yaml` | Image size, normalisation statistics, train-time flip, inference batch size, training hyperparameters, checkpoint path |
| `scoring.yaml` | Palette extraction, colour thresholds, formality levels, season tables, harmony and proportion limits, subscore weights, issue penalties, feature spec |
| `roast.yaml` | Roast persona with its safety block, and the per-request prompt template |

## Environment variables

| Variable | Effect |
|---|---|
| `RMO_ROOT` | Repository root, for cases where `pyproject.toml` cannot be found by walking up |
| `RMO_DATA_ROOT` | Data tree, for example a read-only mount |
| `RMO_SOURCE_DIR` | Default for `download_data.py --from` |
| `GEMINI_API_KEY` | Enables `gemini_roaster` |
| `RMO_GEMINI_MODEL` | Overrides the default Gemini model |

The last two may be set in `.env` at the repository root. `.env.example` is committed as a template
and `.env` is gitignored.

## Project layout

```
configs/            stage configuration in YAML
data/
  fixtures/         committed replay corpus, images and safety probes
  processed/        outfit table, committed splits, prediction exports
  raw/              staged dataset, gitignored
docs/CONTRACTS.md   record, invariant and registry contracts
models/             trained checkpoints and fitted bundles, gitignored
notebooks/          data acquisition and exploratory analysis
results/
  figures/          exploratory analysis plots
  metrics/          evaluation artefacts
scripts/            command line wrappers
src/rmo/
  config.py         cached YAML loading
  fixtures.py       fixture corpus loading keyed by image_id
  imaging.py        input normalisation to RGB images
  paths.py          filesystem layout, atomic writes, hashing, git identity
  pipeline.py       stage registry, OutfitRoaster, CLI
  schemas.py        boundary records and enumerations
  splits.py         group-aware split construction and loading
  data/             download, annotation parsing, table build, preflight
  eval/             metrics envelope, perception, scoring and roast evaluation,
                    prediction export
  perception/       base, dummy, CNN, CNN training, VLM, postprocessing,
                    mask enrichment
  scoring/          base, dummy, rules, learned scorer, scorer training, pairs,
                    missingness guard, colour theory, palette, features
  roast/            base, dummy, rules, Gemini, safety classifier
tests/              pytest suite
```

Asking for a path never creates it. Directory creation is an explicit call, and every artefact write
goes through an atomic temp-and-replace helper.

## Testing

```powershell
pytest -q
```

The suite covers 39 modules spanning schemas, contracts, the registry, the dataset pipeline, split
integrity, scoring rules, colour theory, palette extraction, feature contracts, pair construction,
the missingness guard, the learned scorer, CNN and scorer training, roast generation, the safety
classifier, evaluation metrics, path helpers and the gitignore negation rules. It uses no network
and no GPU. Tests requiring the optional torch, transformers or timm packages skip themselves when
those are absent. Gemini tests run against a stub client, so they need no API key and no network
access.

## Library use

```python
from rmo.pipeline import OutfitRoaster, create

roaster = OutfitRoaster(
    perception=create("dummy_perception"),
    scorer=create("rule_scorer_v1"),
    roaster=create("rule_roaster"),
)
description, score, roast = roaster.run("data/fixtures/images/fixture_000.png")
print(roast.roast)
```

Any stage left as `None` resolves to its registered dummy. `OutfitRoaster.run` accepts a path, a
`PIL.Image`, or an HWC uint8 numpy array, and returns the three records as a tuple. To add a model,
subclass the relevant base class, give it a unique `name`, and call
`rmo.pipeline.register(name, cls)` from the stage package's `__init__.py`.

## License

Released under the MIT license, as declared in `pyproject.toml`. The DeepFashion-MultiModal dataset
is covered by its own non-commercial research license and is never redistributed here.


# Contracts

Three records cross the stage boundaries, and every model on either side of a boundary agrees on
them. They are defined in [`src/rmo/schemas.py`](../src/rmo/schemas.py); this page covers what the
code cannot state on its own.

## The records

| Record | Describes | Produced by | Committed as |
|---|---|---|---|
| `OutfitDescription` | What is worn in one photograph | `PerceptionModel.predict` | `data/fixtures/outfit_descriptions.jsonl` |
| `OutfitScore` | How well it works, and what is wrong | `ScoringModel.score` | `data/fixtures/outfit_scores.jsonl` |
| `RoastOutput` | The text the product returns | `RoastGenerator.generate` | `data/fixtures/roast_outputs.jsonl` |

`Garment`, `SubScores` and `Issue` are nested inside them and are never exchanged on their own.

## Adding to a schema

Anyone may add a field, provided it has a default, or add a member to an enum. Removing, renaming
or tightening anything already there breaks every record written before the change and needs
agreement first.

`tests/test_schema_compat.py` replays `data/fixtures/golden_v1.0.0.jsonl` through today's
`OutfitDescription` and fails if a record stops parsing or stops reproducing any value it was
written with. Extra keys in the reparsed record are ignored, so an addition passes and a removal
does not. When it fails, the field it names is the one to restore — regenerating the golden corpus
hides the breakage instead of fixing it.

## What every record carries

All three top-level records carry the same four fields:

| Field | Meaning |
|---|---|
| `image_id` | The photograph the record is about, identical across all three |
| `provenance` | Where the attributes came from |
| `source_model` | The `name` of the model that produced the record |
| `schema_version` | `SCHEMA_VERSION` at the time it was written |

## Provenance

`Provenance` is `gt` for a human annotation, `predicted` for a model output and `fixture` for a
record in the committed corpus.

`OutfitDescription` defaults to `predicted`. `OutfitScore` and `RoastOutput` have no default and
must state theirs, and each stage copies its input's value forward rather than choosing one. A
fixture description therefore cannot pick up a score that claims to be predicted, and the reverse
cannot happen either.

## Garment refs

`Garment.ref` may be left blank, and `OutfitDescription`'s validator then numbers it per slot, so
the second upper garment becomes `upper_1`. Refs are unique within a description, and they are the
only stable handle a score or a roast may cite — garment order is not one, because a model may list
garments differently.

## Invariants across records

No single model can enforce these, so they belong here and are checked in
`tests/test_contracts.py`:

- every `Issue.garment_refs` entry is a ref of the description that was scored;
- every `RoastOutput.grounded_garments` entry is a ref of that same description;
- `image_id` is identical on the description, the score and the roast.

## Registering a model

A stage package registers its own models when it is imported, so adding one never means editing
`src/rmo/pipeline.py`:

```python
from rmo.pipeline import register
from rmo.scoring.rules import RuleScorer

register(RuleScorer.name, RuleScorer)
```

The class is registered, not an instance, so `create(name)` returns a fresh model and nothing is
built at import time. A name may be taken once; registering it twice raises.

`OutfitRoaster(perception, scorer, roaster)` accepts any three stage instances and falls back to the
registered dummy for each one left out.

## One outfit end to end

```powershell
python -m rmo.pipeline --image data/fixtures/images/fixture_000.png
```

writes one `RoastOutput` to stdout:

```json
{"image_id":"fixture_000","roast":"Safe, clean and about as surprising as a glass of tap water, but nothing here is actually wrong.","suggestions":["Swap the white sneakers for tan leather to break the top-to-toe repeat.","A woven belt would give the waist something to do."],"tone":"gentle","grounded_garments":["upper_0","footwear_0"],"safety_flags":[],"provenance":"fixture","source_model":"fixture","schema_version":"1.0.0"}
```

`grounded_garments` names `upper_0` and `footwear_0`, which are refs of the description that
produced it, and `provenance` is `fixture` because the description it came from was. Add `--json` to
see that description and its score alongside the roast.

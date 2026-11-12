# openadmet-cyp

Code for the OpenADMET CYP Inhibition Blind Challenge.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e .   # installs requirements.txt and registers the CLI entry points (data, train-model, evaluate-models, ...)
```

Folders default to `data/`, `data/features/` and `results/`. To put them elsewhere, copy `.env.example` to `.env` in the
working directory and set `DATA_FOLDER`, `FEATURE_FOLDER` (default `<DATA_FOLDER>/features`) and `RESULTS_FOLDER`. A
variable already set in the shell wins over the file, and a command-line flag such as `--data-dir` wins over both.

## Data

```bash
data list                  # every known dataset with status, rows, size, and what it trains as
data fetch NAME [NAME...]  # fetch by name or alias (or `all`), then derive the model-ready table
data fetch NAME --redownload   # fetch the raw files again instead of reusing what is here
data remove NAME           # delete a dataset's raw and derived files
data alias NAME ALIAS      # short alias, stored in data-alias.csv
download-data              # same as `data fetch all`
```

Fetching reuses raw files that are already on disk; `--redownload` goes back to the source. The derive step runs
either way, so changing a reshape and re-running `data fetch NAME` rebuilds the model-ready table without
touching the network. That matters most for `p450_family`, whose fetch crawls UniProt, ChEMBL, and BindingDB for
hundreds of proteins, and for `pubchem_aid1851`, whose raw table is ~24 MB.

Each dataset lives under `src/data_tools/dataset_defs/` and holds everything it needs: its source URL, its `fetch`
step, an optional `derive` step, the files each writes, and the training heads the result offers (`--datasets
NAME:ISO`). A simple dataset is one module (`train.py`); one that brings its own crawling or reshaping code is a
directory holding that code next to the spec (`pubchem_aid1851/`, `p450_family/`); `metabolism/` just groups
related datasets. Adding a dataset means adding a module there and changing nothing else —
`data_tools.datasets.registry()` discovers it, `data list` shows it, and `--datasets` accepts it. See
`src/data_tools/dataset_spec.py` for the fields.

`pubchem_aid1851` yields per-isoform auxiliary heads; `p450_family` (ChEMBL + BindingDB, Pfam PF00067) yields one
`{gene}_{readout}_{source}` head per protein per source (IC50 only by default, so `{gene}_pIC50_{source}`).

### Structural identity

Sources spell the same molecule differently (salt vs free base, explicit vs
implicit stereo), so raw SMILES comparison under-reports overlap. External sources are
joined, and every overlap measurement keyed, on the **InChIKey connectivity block**
(first 14 characters), which is invariant to stereochemistry, protonation and
isotope labelling. The in-house sources (same molecules) are collapsed on the full
InChIKey so stereoisomers stay distinct. See `data_tools.standardize`.

## Training

`train-model` trains any supported model on exactly the datasets you list.
Pick the model with `--model`:

| `--model` | What it trains |
|---|---|
| `chemprop` | Chemprop D-MPNN from scratch |
| `chemeleon` | Chemprop started from the CheMeleon foundation weights |
| `unimol` | Uni-Mol fine-tuned (needs `unimol_tools`; uses the GPU if available, `--cpu` to force CPU) |
| `atom_jepa` | Atom-JEPA `molecules` encoder fine-tuned with an MLP head (needs `atom-jepa`; GPU recommended) |
| `atom_jepa_bio` | same, started from the Atom-JEPA-Bio encoder (`molecules` continued on PCBA bioassays) |

```bash
train-model --list-datasets                     # selectable sources
train-model --model chemprop --datasets challenge tdc_hlm pubchem:CYP1A2 bindingdb:CYP1A2,CYP2C8 --loss-function mae
train-model --model chemeleon --datasets challenge pubchem:CYP1A2
train-model --model unimol --datasets challenge pubchem:CYP1A2 --learning-rate 2e-4
# configs x learning rates x seeds: repeat --datasets, add --n-seeds N (or --seeds 0 1 2) and, for unimol, several
# --learning-rate values; one model per combination
train-model --model unimol --n-seeds 3 --learning-rate 1e-2 1e-3 1e-4 \
    --datasets challenge \
    --datasets challenge pubchem:CYP1A2 bindingdb:CYP2C8 \
    --datasets pubchem:CYP1A2 bindingdb:CYP2C8
# pretrain on stability, then train on the ADMET data plus auxiliary heads (chemprop/chemeleon)
train-model --model chemprop --stage tdc_hlm tdc_rlm --stage challenge pubchem:CYP1A2
# same thing, with per-stage settings and an encoder-size sweep
train-model --model chemprop --pretrain tdc_hlm --finetune challenge pubchem:CYP1A2 \
    --stage-epochs 30 50 --stage-freeze-epochs 0 5 --stage-lr-warmup-shape linear cosine \
    --stage-loss mae huber --huber-delta 0.5 --hidden-dims 300 600 --mpn-dropouts 0 0.1
```

`--datasets` takes any name from `train-model --list-datasets` (`challenge`, `pubchem`,
`bindingdb`, `tdi`, `tdc_hlm`, ...), as `NAME[:ISO,ISO]`; heads are named
`{isoform}-{source}`. Only what you list is trained on: `--datasets challenge` is the
challenge labels alone, `--datasets challenge pubchem` is both, and `--datasets pubchem`
is the auxiliary data alone. The challenge data is still loaded in every case,
since it provides the pinned train/val split and the validation molecules; with
no `challenge` in the config its labels are just not trained on. For `unimol`, each
challenge head is then scored with the same-isoform auxiliary head, so that RMSE carries a
cross-assay calibration offset. Each head keeps its own scale and missing labels
are masked in the loss. With several runs, outputs go to
`<output-dir>/<config>/seed<N>` (default `results/chemprop/`, `results/chemprop/chemeleon/`, `results/unimol/`, `results/atom_jepa/` or `results/atom_jepa_bio/`). Shared flags:
`--target-columns`, `--split-strategy`, `--split-seed`, `--epochs`, `--seed`,
`--output-dir`.

`atom_jepa` and `atom_jepa_bio` early-stop whichever datasets are listed: the labelled training
molecules are split by scaffold into a fit set and a holdout (`--holdout-frac`, default 0.1),
training stops after `--patience` epochs (default 5) without a better holdout masked MAE, and the best
epoch's weights are kept. The pinned challenge val split is only scored, never used to pick the epoch.
Other flags: `--learning-rate` (swept like seeds), `--head-lr-mult`, `--batch-size`, `--workers`
(conformer generation). Each run writes `encoder.pt` (an `atom_jepa[...]` / `atom_jepa_bio[...]` input)
and `head.pt` under `<config>/lr_<lr>_seed_<N>/`.

Chemprop training options (`src/models/chemprop_ext.py` patches them into
`chemprop train`; `train-model` runs chemprop through it):

| Flag | Effect |
|---|---|
| `--pretrain A B [--pretrain C] --finetune X Y` | Pretrain stages, then the evaluated final stage (shorthand for repeated `--stage`) |
| `--stage-epochs`, `--stage-warmup-epochs`, `--stage-max-lr`, `--stage-lr-warmup-shape`, `--stage-lr-decay`, `--stage-freeze-epochs`, `--stage-loss` | Per-stage values, one per stage or one for all |
| `--loss-function huber --huber-delta D` | Huber loss; `D` in pIC50 units |
| `--label-noise S`, `--label-shrink A` | Training-only target smoothing: Gaussian jitter, or pull toward the batch mean per head |
| `--lr-warmup-shape linear\|cosine\|exp\|none`, `--lr-decay exp\|cosine\|linear\|constant` | Learning-rate schedule shape (chemprop's `--warmup-epochs`, `--init-lr`, `--max-lr`, `--final-lr` still apply) |
| `--freeze-epochs N` | Keep the message-passing encoder frozen for the first `N` epochs |
| `--mpn-dropout P`, `--ffn-dropout P` | Dropout in message passing / the prediction head separately |
| `--hidden-dims`, `--depths`, `--mpn-dropouts` | Encoder-size sweep: every combination trains, times seeds. Later stages inherit the first stage's encoder size |

Checkpoints trained with Huber need `models.chemprop_losses` importable to load.
Censored losses (`--loss-function bounded-mse`) need `<`/`>` in the label CSV;
none of our datasets carry that yet.

`--split-strategy` defaults to `scaffold`, the pinned split every recorded result
is reported against. `blind_matched` instead makes the held-out slices imitate the
blind set's regime — 75% novel scaffolds rather than 100% — which scores about
0.013 lower and is **not** comparable to the pinned numbers. It is a diagnostic for
how much the pinned split understates performance, not a reporting axis. In the
three-model comparison that produced the 0.013 figure (average shift -0.0134, range
-0.0095 to -0.0204) the model ranking did not change.

Chemprop-based models pass unknown flags (`--batch-size`, `--accelerator`,
`--task-weights`, ...) through to `chemprop train`. `--input NAME [NAME ...]` adds
any registered representation (see Representations) as extra molecule features for
every stage: it is computed per molecule, cached in `data/features/`, and passed to
chemprop as `--descriptors-path` (non-finite values become 0 with a warning; chemprop
scales them by default, `--no-descriptor-scaling` turns that off). To train on a ready table
instead, `--data-path <table.csv> --target-columns <heads>`.
On SCC, `scripts/train_model.qsub` forwards its arguments to `train-model`. To add a
model, write a `Backend` in `src/models/training/` and register it in `BACKENDS`.

### Other commands

Every command takes `--help`.

```bash
evaluate-models      # train/val metrics for the sklearn-style models; see Representations and Ensembling
plot-results         # redraw the evaluate-models plots from a saved --predictions-out CSV
foundation-model     # list/alias trained chemprop, unimol and Atom-JEPA runs and generate their feature caches, plus the pretrained chemeleon, unimol, atom_jepa and atom_jepa_bio (--list, --paths, --generate-cache)
motif-report         # generate reports/pharmacophore_features.md from the pharmacophore motif table
list-runs            # previous train-model / evaluate-models runs (--train-model, --eval, -n N, or an id for one run)
```

`train-model` (one log per seed/config of a sweep) and `evaluate-models` write their output to
`results/runs/<id>__<train|eval>__<flags>.log` while still printing it. The id counts up, and the filename lists the
flags that differ from their default (`key=value`, `+` between list items). Each log starts with the full command and git
commit and ends with a `# finished:` line holding status and duration.


## Representations

`evaluate-models` takes `--input` with one or more of the
names below; several names are hstacked into one feature matrix.

| `--input` | Class | Dim | What it is |
|---|---|---:|---|
| `rdkit` | `RDKitDescriptors` | 217 | RDKit 2D physicochemical descriptors |
| `chemeleon` | `CheMeleon` | 2048 | fixed pretrained CheMeleon embedding; checkpoint downloads on first use |
| `morgan` | `MorganFingerprint` | 2048 | Morgan/ECFP4 bits, radius 2 |
| `count_morgan` | `CountMorganFingerprint` | 2048 | same generator, substructure counts |
| `maccs` | `MACCS` | 167 | MACCS structural keys |
| `mordred` | `MordredDescriptors` | 1613 | Mordred 2D descriptors (`mordredcommunity`); missing values encoded as 0 |
| `xtb_solv_v1` | `XTBSolvation` | 9 | GFN2-xTB (`tblite`) ALPB water/octanol solvation, xTB logP, heuristic logD at pH 7.4; slow, see note below |
| `xtb_elec_v1` | `XTBElectronic` | 24 | Boltzmann-averaged GFN2-xTB FMO energies, charges, SASA, shape, flexibility; shares conformers with `xtb_solv_v1` |
| `cyp_mech_v1` | `CYPMechanisticBlock` | 30 | Mechanistic block: N/acid group counts, rule-based pKa and pH 7.4 protonation state, basic-N-to-aryl topological distances, CYP2D6 pharmacophore flags. Ported from Lizard Wizard Gizzard, see Acknowledgements |
| `shape` | `ShapeDescriptors` | 11 | Deterministic RDKit 3D shape descriptors |
| `erg` | `ErGFingerprint` | 315 | RDKit extended reduced graph (ErG) pharmacophore fingerprint |
| `unimol` | `UniMol` | 512 | Uni-Mol embedding (pretrained, or a fine-tuned model); needs `unimol_tools` |
| `atom_jepa` | `AtomJEPA` | 256 | fixed pretrained Atom-JEPA `molecules` encoder (3D, conformer generated by the package); needs `atom-jepa`, GPU recommended |
| `atom_jepa_bio` | `AtomJEPABio` | 256 | fixed pretrained Atom-JEPA-Bio encoder (`molecules` continued on PCBA bioassays); same requirements |
| `cyp_heme_v2` | `HemeCoordinationMotifs` | 13 | SMARTS counts for nitrogens able to ligate the heme iron, plus a steric-accessibility split |
| `cyp_isoform_v2` | `IsoformPreferenceDescriptors` | 10 | Physicochemical descriptors chosen for the four scored isoforms, each cited |
| `cyp_tdi_v2` | `MechanismBasedAlerts` | 6 | Structural alerts for time-dependent inactivation |

The xTB blocks embed up to 10 ETKDG/MMFF conformers per molecule and run three
xTB single points each. Cost is seconds per molecule, so the first run is the
expensive one (results are cached in `data/features/`). Worker processes come from
`XTB_NJOBS`, else `NSLOTS` (SGE), else the CPU count. Request both blocks in one
call (`--input xtb_solv_v1 xtb_elec_v1`) to compute each molecule once.

```bash
evaluate-models --input morgan                          # one representation
evaluate-models --input count_morgan rdkit --split butina --seed 0
evaluate-models --input all                             # every representation on its own, one table
evaluate-models --models ridge --input morgan --predictions-out results/validation/morgan.csv
```

`--predictions-out <path>` optionally saves one UTF-8 long-format CSV containing
validation predictions for every selected model, input set, and target (when blind-set
predictions are computed, via `--predict-blind` or `--plots train-test`, it also writes a
sibling `<stem>.blind.csv`, `molecule,isoform,model,input,split,seed,y_pred`, which
`plot-results` reads; `--predict-blind` separately writes one
`results/<model>_<input>_blind_predictions.csv` per model and input):

```text
smiles,isoform,model,input,split,seed,y_true,y_pred
```

`smiles` preserves the source `SMILES` string; `isoform` is the full target
column name and `model` is its registry name. Combined inputs use their ordered
names joined with ` + ` (for example, `count_morgan + rdkit`); `--input all`
produces separate groups. `split` records the executed strategy (`random`,
`scaffold`, `butina`, or `blind_matched`), and `seed` is the **split seed only**. This evaluator
does not use the pinned split column. With `--nseeds 1` models keep their own
default seeds; with `--nseeds N` they are seeded `--seed` to `--seed + N - 1`, but
the CSV still records only `--seed`. Record the command alongside exports, including
`--val-split`, `--butina-cutoff`, and the data source, which are not CSV columns.

Each model predicts validation once, and those exact values are scored and
exported. Non-finite feature values are first set to 0 (with a warning on stderr), then
`drop_nan_rows` drops rows with a non-finite label, selecting features, labels,
and SMILES together and retaining their order. Every surviving
row is saved, even if another representation drops that molecule. Duplicate
source SMILES within a `(model, input, isoform, split, seed)` group cause export
to fail with an explicit error before writing; rows are never deduplicated.
Repeated selections that duplicate the same group also fail. Without the flag,
evaluation retains its existing behavior, including duplicate rows.

Use ignored `results/` paths for generated CSVs. Missing parent directories are
created. Existing files are refused (exclusive creation, no overwrite or append).
Write errors report the destination and underlying error and exit unsuccessfully;
an I/O failure after creation can leave a partial file, which must be removed or
given a new destination before retrying. Cross-run SMILES inner joins and
prediction/residual correlations are a separate next task.

### Evaluating an embedding

`representations --evaluate <name>... | all` scores embeddings on a train/val
split (`--split` default `scaffold`, plus `--val-split`, `--seed`,
`--butina-cutoff`, `--targets`, `--max-samples`) using the feature cache. It prints
two tables; label metrics are averaged over the targets that have data.

| Metric | Column | Reads as |
|---|---|---|
| ridge probe val R2 | Probe R2 | higher is better |
| 5-NN regression val R2 | kNN R2 | higher is better |
| Tanimoto neighbour recall@10 | Recall | embedding 10-NN overlap with Morgan 10-NN |
| effective rank | EffRank | low means dimensional collapse |
| roughness (1-NN label diff / random-pair diff) | Rough | below 1 means neighbours share labels |
| val-to-train NN distance vs train leave-one-out | NN ratio, Out | 1.0 and 0.05 mean val looks like train |
| train-vs-val classifier AUROC | AUC | 0.5 indistinguishable, 1.0 separable |
| RBF MMD permutation p-value | MMD p | small means the distributions differ |
| Spearman of val NN distance vs probe error | Err rho | positive means distance predicts failure |

The first row of each distance table is the same NN ratio and Out measured on
Morgan Tanimoto, the baseline for how hard the split is on its own.

A third table, on by default (`--no-test` skips it), measures the blinded
`data/test.csv` against all of `train.csv` with the label-free metrics (NN ratio,
Out, AUC, MMD p), so the real test shift can be compared with the validation shift. NN ratio below 1 means
test sits closer to train than train does to itself, while a high AUC with it
means test is still a different distribution. Features are
standardized with train statistics first, and the pairwise metrics subsample to
`--max-samples` molecules.

Unparsable SMILES give an all-NaN row; `evaluate-models` zeroes non-finite feature values (a molecule is then dropped only for a missing label), while `representations --evaluate` drops non-finite rows. Features
are cached per representation in `data/features/{name}.npy`, keyed on SMILES, so
each molecule is featurized once across every split and seed. To add one,
subclass `Representation` in `src/representations/` and append it to
`INPUT_REGISTRY` in `src/data_tools/inputs.py`.

## Models

Registered models accept precomputed feature matrices through `fit(X, y)` and
`predict(X)`. The evaluator applies `models.utils.drop_nan_rows` before fitting
and scoring. The three ensemble wrappers do not add filtering, feature generation,
scaling or PCA. Select models with `evaluate-models --models <name> ...`.

| Name | Estimator | Default settings |
|---|---|---|
| `decision_tree` | sklearn DecisionTreeRegressor | depth=5, min split=2, min leaf=1, seed=42 |
| `ridge` | sklearn StandardScaler + Ridge | alpha=1.0 |
| `random_forest` | sklearn RandomForestRegressor | 200 trees, unlimited depth, min leaf=1, all features |
| `lightgbm` | LightGBM LGBMRegressor | 200 trees, rate=0.05, 31 leaves, unlimited depth, min child=20 |
| `xgboost` | XGBoost XGBRegressor | 200 trees, rate=0.05, depth=6, hist trees |
| `tabpfn` | TabPFN TabPFNRegressor | n_estimators=auto, temperature=auto, device=auto, seed=42 |

The three ensemble wrappers default to `random_state=42` and `n_jobs=1`. Random Forest runs on CPU; LightGBM and XGBoost default to `device="auto"` (CUDA when a GPU is visible, else CPU).
Random Forest uses `criterion=squared_error`, `bootstrap=True` and `max_features=1.0`.
LightGBM uses `boosting_type=gbdt`, `objective=regression`, `verbosity=-1`. Its unlimited depth is `max_depth=-1`.
XGBoost uses `objective=reg:squarederror`, `booster=gbtree`, `tree_method=hist`,
`subsample=1.0` and `colsample_bytree=1.0`.

Constructors expose tree count, complexity controls, seed and worker count;
the boosting wrappers also expose learning rate. The evaluator uses constructor
defaults. These are baseline settings, not tuned CYP results.

TabPFN does no gradient training: `fit` stores the training rows and the
transformer conditions on them at predict time. Its constructor exposes
`n_estimators`, `softmax_temperature`, `ignore_pretraining_limits`, `device`,
`model_path`, `random_state` and `n_components`; it defaults to `device="auto"` rather than CPU,
since the others are tree models and this one is a transformer. By default
it first reduces the features to `n_components=256` with a PCA step (`None` disables it). Set
`ignore_pretraining_limits=True` to feed it representations wider than the
pretraining feature limit.

`tabpfn` is always registered. Its constructor defers `import tabpfn`, so listing it
costs nothing, but the package must be installed to construct it. It runs with its defaults, untuned.

## Splits

`data_tools.load.load_data` supports `random`, `scaffold` (Bemis-Murcko),
`butina` (Tanimoto cluster) and `blind_matched` train/val splits; `evaluate-models`
exposes them as `--split` and `--seed`. `blind_matched` makes the validation slice
imitate `data/test.csv` (see Training) and needs that file present.

## Ensembling

`evaluate-models` ensembles across folds, seeds, models and inputs. Everything
below needs `--nfolds` except plain averaging.

```bash
evaluate-models --models ridge random_forest lightgbm --input morgan --input rdkit \
    --nfolds 5 --cv-split scaffold --nseeds 3 --ensemble nnls \
    --folds-out results/folds_scaffold5.csv
```

- `--nfolds N`, `--cv-split random|scaffold|butina`: k-fold CV on the training
  split; `--folds-out` saves each training molecule's fold to a new CSV.
- `--nseeds N`: seed averaging per model (models without `random_state` train once).
- `--ensemble average|min|max|nnls|caruana|ridgecv|elasticnetcv`: how models are combined into
  an `ensemble[...]` row. `caruana` is greedy selection with replacement
  (nonnegative weights summing to one, no intercept). The stackers learn per-target weights from out-of-fold
  predictions (`ridgecv` and `elasticnetcv` need `--nfolds >= 3`), print the
  weights, shares, intercept and chosen hyperparameters, and score the stack
  with leave-one-fold-out refits. Models are still averaged over folds and seeds.
- With several input sets (`--input` repeated), each target also gets an
  `all inputs` ensemble over every (model, input) pair; `--no-input-ensemble`
  turns that off.
- `--cv-filter r2 RAE`: before ensembling, drop models whose out-of-fold R² is below
  0.15 and models whose validation RAE is more than 2x their out-of-fold RAE
  (the leakage check). If none pass, all are kept with a warning. This is the
  default with `--nfolds`; `--cv-filter none` disables it.
- Models whose validation predictions correlate above 0.95 (Pearson, complete
  linkage) are averaged into one before ensembling; `--no-cv-deduplication`
  disables this. It runs after the filters.
- Every filter and deduplication decision is logged to stderr and summarised in an
  `ensemble filters` table with each model's scores, kept/dropped status and cluster.

Stackers are fitted only on out-of-fold predictions, never on predictions the base
models saw in training. Validation informs selection, so validation metrics are
development results.

Feature models do not use auxiliary target columns, but can use embeddings from
auxiliary-trained checkpoints (`chemprop[...]`, `unimol[...]`, `atom_jepa[...]`, `atom_jepa_bio[...]` inputs).
Checkpoints must not have been trained on held-out labels. Record their resolved
paths and hashes, keep weights immutable, and use fresh caches when changing them.

## Acknowledgements

`cyp_mech_v1` (`src/representations/cyp_mech.py`) is ported from the open-code
submission of the Lizard Wizard Gizzard team,
[doctawho42/CYPChallenge](https://github.com/doctawho42/CYPChallenge) (MIT License,
Copyright (c) 2026 the CYPChallenge contributors), at commit `bf018b8`. The SMARTS
tables, pKa rule and column definitions are theirs; see `THIRD_PARTY_NOTICES.md`.

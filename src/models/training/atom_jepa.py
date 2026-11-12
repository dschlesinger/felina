"""Atom-JEPA backend: fine-tune a pretrained Atom-JEPA encoder jointly on the listed datasets.

One model per (learning rate, seed) with every head at once. Missing labels are masked in the loss, so molecules
measured by only one source still train that head. `--model atom_jepa` starts from the `molecules` encoder and
`--model atom_jepa_bio` from the `bio` encoder (continued on PCBA bioassays).

Whatever datasets are listed, validation works the same way: the labelled training molecules (those with a label
on at least one selected head) are split by scaffold into a fit set and a holdout. The model trains on the fit set,
early-stops on the holdout's masked MAE, and the best epoch's weights are kept. The pinned challenge val split is
never used for epoch selection; it is only scored and written out, like the other backends.

  train-model --model atom_jepa --datasets challenge pubchem:CYP1A2
  train-model --model atom_jepa_bio --datasets pubchem:CYP1A2 bindingdb:CYP2C8

Each run writes `encoder.pt` (loadable as the `atom_jepa[path]` / `atom_jepa_bio[path]` input) and `head.pt`.
"""

import argparse
import dataclasses
import multiprocessing
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from data_tools.merge_datasets import parse_dataset_specs
from data_tools.paths import RESULTS_DIR
from models.training.base import Backend
from models.training.common import prepare_joint_split, write_val_outputs
from models.training.unimol import check_cuda, set_seed

_CLIP_NORM = 1.0
_SAMPLES: dict[str, dict | None] = {}  # conformer cache shared by every run of one process


def _conformer(smiles: str) -> dict | None:
    """One molecule's Atom-JEPA sample (atomic numbers, coordinates), or None if no conformer can be built."""
    from atom_jepa import to_sample

    try:
        return to_sample(smiles)
    except (ValueError, RuntimeError, TypeError):
        return None


def conformers(smiles: list[str], workers: int) -> dict[str, dict | None]:
    """Samples for every SMILES, generating only the ones not yet cached. Failures map to None."""
    todo = [s for s in dict.fromkeys(smiles) if s not in _SAMPLES]
    if todo:
        if workers > 1:
            with multiprocessing.get_context("spawn").Pool(workers) as pool:  # spawn: no fork after CUDA/OpenMP use
                results = list(tqdm(pool.imap(_conformer, todo, chunksize=64), total=len(todo), desc="Conformers", unit="mol"))
        else:
            results = [_conformer(s) for s in tqdm(todo, desc="Conformers", unit="mol")]
        _SAMPLES.update(zip(todo, results))
    return {s: _SAMPLES[s] for s in smiles}


def split_holdout(df: pd.DataFrame, fraction: float, seed: int) -> np.ndarray:
    """Boolean mask of the holdout rows: whole Murcko scaffolds, in seeded random order, until `fraction` is reached.

    Rows sharing an InChIKey connectivity block always share a scaffold, so no block straddles the split.
    Acyclic molecules are their own groups.
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold

    if not 0 < fraction < 1:
        raise ValueError(f"holdout fraction must be in (0, 1), got {fraction}")
    block_scaffold: dict[str, str] = {}
    for block, smiles in zip(df["inchikey_block"], df["SMILES"]):
        if block not in block_scaffold:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
            except (ValueError, TypeError):
                scaffold = ""
            block_scaffold[block] = scaffold or f"acyclic:{block}"
    groups: dict[str, list[int]] = {}
    for position, block in enumerate(df["inchikey_block"]):
        groups.setdefault(block_scaffold[block], []).append(position)

    target = max(1, round(fraction * len(df)))
    order = np.random.default_rng(seed).permutation(sorted(groups))
    holdout = np.zeros(len(df), dtype=bool)
    for key in order:
        if holdout.sum() >= target:
            break
        holdout[groups[key]] = True
    return holdout


def masked_abs_error(pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head summed absolute error and label count over the non-NaN labels; pred and y are (batch, heads)."""
    mask = ~torch.isnan(y)
    err = (pred - torch.nan_to_num(y)).abs() * mask
    return err.sum(dim=0), mask.sum(dim=0)


def _batches(n: int, batch_size: int, rng: np.random.Generator | None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def _forward(model, head, smiles: np.ndarray, idx: np.ndarray, samples) -> torch.Tensor:
    return head(model(model.collate([samples[s] for s in smiles[idx]])))


def _holdout_mae(model, head, smiles: np.ndarray, y: torch.Tensor, samples, batch_size: int) -> float:
    """Mean over heads of the masked MAE (standardized labels); heads without a holdout label don't count."""
    model.eval()
    head.eval()
    err = torch.zeros(y.shape[1])
    count = torch.zeros(y.shape[1])
    with torch.no_grad():
        for idx in _batches(len(smiles), batch_size, None):
            e, c = masked_abs_error(_forward(model, head, smiles, idx, samples).float().cpu(), y[idx])
            err += e
            count += c
    seen = count > 0
    return float((err[seen] / count[seen]).mean())


def _predict(model, head, smiles: np.ndarray, samples, batch_size: int) -> np.ndarray:
    model.eval()
    head.eval()
    out = []
    with torch.no_grad():
        for idx in _batches(len(smiles), batch_size, None):
            out.append(_forward(model, head, smiles, idx, samples).float().cpu().numpy())
    return np.concatenate(out)


def _snapshot(*modules: nn.Module) -> list[dict[str, torch.Tensor]]:
    return [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in modules]


class AtomJEPABackend(Backend):
    requires_datasets = True
    defaults = {"epochs": 30}

    def __init__(self, name: str, variant: str) -> None:
        super().__init__(name)
        self.variant = variant
        self.results_dir = RESULTS_DIR / name

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--learning-rate", nargs="+", type=float, default=[1e-4], help="One or more encoder learning rates; each is swept like a seed (default: 1e-4)"
        )
        parser.add_argument("--head-lr-mult", type=float, default=10.0, help="Head learning rate as a multiple of the encoder's (default: 10)")
        parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay (default: 0.01)")
        parser.add_argument("--batch-size", type=int, default=32, help="Molecules per batch (default: 32)")
        parser.add_argument("--patience", type=int, default=5, help="Stop after this many epochs without a better holdout MAE (default: 5)")
        parser.add_argument("--holdout-frac", type=float, default=0.1, help="Fraction of the training molecules held out for early stopping (default: 0.1)")
        parser.add_argument("--precision", choices=["auto", "fp32"], default="auto", help="auto = bf16 on supported GPUs, else fp32")
        parser.add_argument("--workers", type=int, default=int(os.environ.get("NSLOTS") or os.cpu_count() or 1), help="Processes for conformer generation (default: $NSLOTS or the CPU count)")
        parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    def validate(self, args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        if not 0 < args.holdout_frac < 1:
            parser.error(f"--holdout-frac must be in (0, 1), got {args.holdout_frac}")
        if args.patience < 1:
            parser.error(f"--patience must be at least 1, got {args.patience}")

    def run(self, args: argparse.Namespace, extra_args: list[str]) -> int:
        try:
            specs = parse_dataset_specs(args.datasets or [], with_challenge=False)
        except ValueError as exc:
            args.parser.error(str(exc))
        set_seed(args.seed)

        train_df, val_df, target_columns, eval_map = prepare_joint_split(specs, args, args.parser)

        # Conformers first: they need no GPU, and a failure on a validation molecule should stop the run early.
        samples = conformers(list(dict.fromkeys([*train_df["SMILES"], *val_df["SMILES"]])), args.workers)
        usable = train_df["SMILES"].map(lambda s: samples[s] is not None).to_numpy()
        if not usable.all():
            print(f"Dropping {int((~usable).sum())} training molecules without a conformer")
            train_df = train_df[usable]
        failed_val = [s for s in val_df["SMILES"] if samples[s] is None]
        if failed_val:
            raise RuntimeError(f"No conformer for {len(failed_val)} validation molecules, e.g. {failed_val[:3]}")

        holdout = split_holdout(train_df, args.holdout_frac, args.seed)
        fit_df, hold_df = train_df[~holdout], train_df[holdout]
        if hold_df[target_columns].notna().sum().sum() == 0:
            raise ValueError("The holdout has no labels on the selected heads; raise --holdout-frac")

        y_fit = fit_df[target_columns].to_numpy(dtype=np.float64)
        labelled = np.isfinite(y_fit)
        count = labelled.sum(axis=0)
        mean = np.where(count > 0, np.where(labelled, y_fit, 0.0).sum(axis=0) / np.maximum(count, 1), 0.0)
        var = np.where(labelled, (y_fit - mean) ** 2, 0.0).sum(axis=0) / np.maximum(count, 1)
        std = np.where(var > 1e-12, np.sqrt(var), 1.0)

        def standardized(frame: pd.DataFrame) -> torch.Tensor:
            return torch.tensor((frame[target_columns].to_numpy(dtype=np.float64) - mean) / std, dtype=torch.float32)

        fit_smiles, hold_smiles = fit_df["SMILES"].to_numpy(), hold_df["SMILES"].to_numpy()
        y_fit_t, y_hold_t = standardized(fit_df), standardized(hold_df)

        use_cuda = torch.cuda.is_available() and not args.cpu
        if use_cuda:
            check_cuda()

        base_dir = args.output_dir or RESULTS_DIR / self.name / args.run_label
        output_dir = base_dir / f"lr_{args.learning_rate}_seed_{args.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Datasets: {', '.join(args.datasets or []) or 'none'}")
        print(f"Heads: {len(target_columns)} ({len(eval_map)} scored)")
        print(f"Fit molecules: {len(fit_df)}, holdout molecules: {len(hold_df)}")
        print(f"Challenge validation molecules (scored only): {len(val_df)}")
        print(f"Encoder: {self.variant}, learning rate {args.learning_rate}, head x{args.head_lr_mult}, seed {args.seed}")
        print(f"Device: {'CUDA' if use_cuda else 'CPU'}")

        from atom_jepa import AtomJEPA as Encoder

        model = Encoder.from_pretrained(self.variant, device="cuda" if use_cuda else "cpu", precision=args.precision, verbose=False)
        model.set_grad_checkpointing(True)
        width = model.embedding_dim
        head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, len(target_columns))).to(model.device)
        optimizer = torch.optim.AdamW(
            [
                {"params": list(model.parameters()), "lr": args.learning_rate},
                {"params": list(head.parameters()), "lr": args.learning_rate * args.head_lr_mult},
            ],
            weight_decay=args.weight_decay,
        )

        best_mae, best_epoch, best_state, wait = float("inf"), 0, _snapshot(model, head), 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            head.train()
            total, seen = 0.0, 0
            for idx in tqdm(list(_batches(len(fit_smiles), args.batch_size, np.random.default_rng(args.seed * 1000 + epoch))), desc=f"Epoch {epoch}", unit="batch", leave=False):
                pred = _forward(model, head, fit_smiles, idx, samples).float()
                err, n = masked_abs_error(pred, y_fit_t[idx].to(pred.device))
                if n.sum() == 0:
                    continue
                loss = err.sum() / n.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], _CLIP_NORM)
                optimizer.step()
                total, seen = total + float(loss) * int(n.sum()), seen + int(n.sum())
            mae = _holdout_mae(model, head, hold_smiles, y_hold_t, samples, args.batch_size * 2)
            improved = mae < best_mae
            print(f"Epoch {epoch}/{args.epochs} train_mae {total / max(seen, 1):.4f} holdout_mae {mae:.4f}{' *' if improved else ''}")
            if improved:
                best_mae, best_epoch, best_state, wait = mae, epoch, _snapshot(model, head), 0
            else:
                wait += 1
                if wait >= args.patience:
                    print(f"Early stopping at epoch {epoch}; best epoch {best_epoch} (holdout_mae {best_mae:.4f})")
                    break

        model.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])

        val_pred = _predict(model, head, val_df["SMILES"].to_numpy(), samples, args.batch_size * 2) * std + mean
        if not np.isfinite(val_pred).all():
            raise RuntimeError("Non-finite predictions returned by Atom-JEPA.")

        self._save(model, head, output_dir, target_columns, mean, std, args)
        write_val_outputs(
            output_dir,
            f"{self.name}_{args.run_label}_lr{args.learning_rate}_seed{args.seed}",
            val_df,
            fit_df,
            dict(zip(target_columns, val_pred.T)),
            eval_map,
            {"learning_rate": args.learning_rate, "seed": args.seed, "best_epoch": best_epoch, "holdout_mae": best_mae, "encoder": self.variant},
        )
        return 0

    @staticmethod
    def _save(model, head, output_dir, target_columns, mean, std, args) -> None:
        """Write encoder.pt (reloadable via `from_pretrained(path)`) and head.pt, then check the encoder reloads."""
        from atom_jepa import AtomJEPA as Encoder

        config = model.config
        state = {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()}
        encoder_path = output_dir / "encoder.pt"
        torch.save(
            {"context_encoder": state, "eqv3_cfg": dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config)},
            encoder_path,
        )
        torch.save(
            {
                "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "target_columns": target_columns,
                "y_mean": mean,
                "y_std": std,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            },
            output_dir / "head.pt",
        )
        # Fail now, not when the weights are first used as an input, if the saved format doesn't round-trip.
        reloaded = Encoder.from_pretrained(str(encoder_path), device="cpu", precision="fp32", verbose=False).encoder.state_dict()
        if reloaded.keys() != state.keys() or any(not torch.allclose(state[k].float(), reloaded[k].float().cpu()) for k in state):
            raise RuntimeError(f"{encoder_path} does not reload to the trained encoder")

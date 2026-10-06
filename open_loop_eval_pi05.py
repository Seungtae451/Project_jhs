# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PI05 CHANGE: OpenPI JAX adaptation; GR00T is not a runtime dependency.
# Upstream: NVIDIA/Isaac-GR00T/gr00t/eval/open_loop_eval.py
# Inspected blob: 4461b84431f33634b953c9e563addd91e20ab9ec
"""
Copy to OpenPI/scripts/open_loop_eval_pi05.py. Run from your OpenPI repo root:

uv run --offline python scripts/open_loop_eval_pi05.py \
  --dataset-path /path/to/dataset_or_parent \
  --model-path /path/to/saved_step \
  --config-name pi05_projection_60d_multi \
  --traj-ids 0 1 --execution-horizon 16 --steps 200

GR00T structure retained: plot_trajectory_results, evaluate_single_trajectory,
ArgsConfig, main and tyro CLI. pi0.5-specific replacements use PI05 CHANGE comments.

--model-path is a JAX saved step with params/ and assets/, not the experiment
parent or original base checkpoint. A params/ path is accepted too.
--save-plot-path has GR00T's meaning: a FIGURE FILE, e.g. ./eval/traj.jpeg.
Multiple episodes get dataset/episode suffixes to prevent overwrites.
Default: /tmp/open_loop_eval/<dataset>/traj_<episode>.jpeg.

Your LOCAL training config defines model architecture, cameras, joint ordering
and action-space transforms. Checkpoint norm stats are used, never recomputed.
--norm-stats-path is an explicit override using EXACT training norm_stats.json.

Use recorded observations, predict chunks, score their first execution_horizon
actions, then advance by that stride. Never pass ground-truth actions to inference.
Unnormalized MSE/MAE are averaged per episode, as in GR00T.
This is NOT closed-loop task-success evaluation. Evaluating training episodes
measures imitation fit, not held-out generalization. Data is never rewritten.

Default offline mode requires existing assets/tokenizer cache (OPENPI_DATA_HOME).
--no-offline opts into normal asset resolution.
--dry-run checks data/transforms/statistics without loading model weights.
--self-test uses only numpy. Full GPU inference needs your own OpenPI environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import dataclasses
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.parse import urlparse

import numpy as np

# PI05 CHANGE: heavy dependencies are loaded lazily; CLI remains tyro.

def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    execution_horizon: int,
    save_plot_path: str,
    # PI05 CHANGE: absolute state and delta action are not comparable.
    plot_state: bool = False,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        traj_id: Trajectory ID
        state_keys: List of state modality keys
        action_keys: List of action modality keys
        execution_horizon: Number of predicted-chunk steps executed per inference
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # PI05 CHANGE: support headless/SSH machines.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys
    fig.suptitle(
        f"Trajectory {traj_id} - State: {', '.join(state_keys)} | Action: {', '.join(action_keys)}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if plot_state and state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        # put a dot every ACTION_HORIZON
        for j in range(0, actual_steps, execution_horizon):
            if j == 0:
                ax.plot(
                    j,
                    gt_action_across_time[j, action_idx],
                    "ro",
                    label="inference point",
                )
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    # PI05 CHANGE: do not silently overwrite an earlier evaluation.
    if Path(save_plot_path).exists():
        raise FileExistsError(f"Plot already exists: {save_plot_path}")
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory


def evaluate_single_trajectory(
    policy,
    loader,
    traj_id: int,
    # PI05 CHANGE: OpenPI TrainConfig replaces embodiment_tag/modality_keys.
    steps=300,
    execution_horizon=16,
    save_plot_path=None,
    plot_state=False,
):
    """Same GR00T evaluation sequence using OpenPI model/data adapters."""
    traj = loader[traj_id]
    traj_length = len(traj)
    actual_steps = traj_length if steps == 0 else min(steps, traj_length)
    logging.info(
        f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})"
    )

    # PI05 CHANGE: recorded observation extraction + policy.infer replace
    # extract_step_data / parse_observation_gr00t / policy.get_action.
    # The SAME range(0, actual_steps, execution_horizon) loop lives in this helper.
    local_args = SimpleNamespace(steps=steps, execution_horizon=execution_horizon,
                                 prompt=loader.prompt)
    gt_action_across_time, pred_action_across_time, state_joints_across_time, _, result = (
        _evaluate_episode_chunks(
            loader.dataset, traj.start, traj.end, loader.repack,
            loader.forward, loader.backward, policy, local_args,
        )
    )
    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    # SAME as GR00T: MSE/MAE across time and action dimensions, unnormalized.
    mse, mae = result["mse"], result["mae"]
    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")
    logging.info(f"state_joints vs time {state_joints_across_time.shape}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=traj_id,
        state_keys=["observation.state"],
        action_keys=list(loader.action_keys),
        execution_horizon=execution_horizon,
        save_plot_path=save_plot_path or f"/tmp/open_loop_eval/traj_{traj_id}.jpeg",
        plot_state=plot_state,
    )
    return mse, mae


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy; GR00T dataclass/tyro pattern."""

    steps: int = 200
    """Max frames per trajectory; PI05 addition: 0 evaluates the whole episode."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """Episode IDs, local to each selected dataset."""

    execution_horizon: int = 16
    """Number of predicted-chunk steps scored per recorded observation."""

    dataset_path: str | None = None
    """PI05 CHANGE: local LeRobot dataset or common parent containing children."""

    model_path: str | None = None
    """PI05 CHANGE: JAX saved step directory with params/ and assets/."""

    denoising_steps: int = 10
    """PI05 CHANGE: sample_actions(num_steps=...), not GR00T action_head."""

    save_plot_path: str | None = None
    """Same as GR00T: figure FILE path, not an output directory."""

    # PI05 CHANGE: OpenPI config replaces GR00T embodiment_tag.
    # GR00T host/port remote inference is omitted; this is local JAX inference.
    config_name: str = "pi05_projection_60d_multi"
    """Exact training config registered in your local OpenPI code."""

    repo_ids: list[str] | None = None
    """PI05 addition: select children of --dataset-path; default all valid children."""

    all_episodes: bool = False
    """PI05 addition: evaluate all episodes instead of --traj-ids."""

    norm_stats_path: str | None = None
    """PI05 addition: EXACT training norm_stats.json if absent from checkpoint."""

    prompt: str | None = None
    """PI05 addition: optional override; otherwise use the dataset task text."""

    offline: bool = True
    """PI05 addition: cache-only OpenPI + HF offline mode; --no-offline opts out."""

    plot_state: bool = False
    """PI05 CHANGE: opt-in only if state and action semantics/units match."""

    dry_run: bool = False
    """PI05 addition: validate data and transforms WITHOUT loading weights."""

    self_test: bool = False
    """PI05 addition: numpy-only regression tests."""


def main(args: ArgsConfig):
    logging.basicConfig(level=logging.INFO)
    if args.self_test:
        self_test()
        return
    if args.dataset_path is None or args.model_path is None:
        raise ValueError("--dataset-path and --model-path are required.")
    if args.steps < 0 or args.execution_horizon < 1 or args.denoising_steps < 1:
        raise ValueError("steps >= 0; execution-horizon and denoising-steps > 0 required.")

    # PI05 CHANGE: Gr00tPolicy -> OpenPI create_trained_policy using
    # local JAX parameters and the exact saved training normalization statistics.
    cfg, data_cfg, Dataset, repack, forward, backward, policy, _, _ = load_runtime(args)
    paths = discover_datasets(Path(args.dataset_path), args.repo_ids)
    all_mse = []
    all_mae = []

    # PI05 addition: run the SAME GR00T single-dataset loop for each child.
    # Do not merge child episode IDs or task mappings.
    for path in paths:
        dataset = Pi05EpisodeLoader(path, Dataset, cfg, data_cfg, repack, forward, backward, args.prompt)
        logging.info(f"Dataset: {path.name}; length: {len(dataset)}")
        traj_ids = sorted(dataset.spans) if args.all_episodes else list(dict.fromkeys(args.traj_ids))
        logging.info(f"Running evaluation on trajectories: {traj_ids}")
        for traj_id in traj_ids:
            # PI05 CHANGE: membership in actual episode IDs, not positional index.
            if traj_id not in dataset.spans:
                raise ValueError(f"{path.name}: episode {traj_id} missing; IDs={sorted(dataset.spans)}")
            if args.dry_run:
                traj = dataset[traj_id]
                packed, gt = prepare_sample(dataset.dataset, traj.start, repack, args)
                check_roundtrip(packed, forward, backward)
                if gt.shape[-1] != cfg.model.action_dim:
                    raise ValueError(f"GT dimension {gt.shape[-1]} != model {cfg.model.action_dim}")
                logging.info(f"DRY RUN OK: {path.name} episode {traj_id}, GT={gt.shape}")
                continue
            logging.info(f"Running trajectory: {traj_id}")

            # PI05 CHANGE: preserve original FILE-path CLI while preventing
            # original multi-episode save_plot_path overwrites.
            if args.save_plot_path:
                save_path = Path(args.save_plot_path).expanduser()
                if not save_path.suffix:
                    raise ValueError("--save-plot-path must be a file, e.g. ./eval/traj.jpeg")
                if len(paths) > 1 or len(traj_ids) > 1:
                    save_path = save_path.with_name(
                        f"{save_path.stem}_{path.name}_traj_{traj_id}{save_path.suffix}"
                    )
            else:
                save_path = Path("/tmp/open_loop_eval") / path.name / f"traj_{traj_id}.jpeg"

            mse, mae = evaluate_single_trajectory(
                policy, dataset, traj_id, steps=args.steps,
                execution_horizon=args.execution_horizon,
                save_plot_path=str(save_path), plot_state=args.plot_state,
            )
            logging.info(f"MSE for trajectory {traj_id}: {mse}, MAE: {mae}")
            all_mse.append(mse)
            all_mae.append(mae)

    if args.dry_run:
        logging.info("Selected input checks passed. Model inference was NOT tested.")
        return
    if all_mse:
        # SAME as GR00T: equal-weight episode average, NOT frame-weighted.
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        logging.info(f"Average MSE across all trajs: {avg_mse}")
        logging.info(f"Average MAE across all trajs: {avg_mae}")
    else:
        logging.info("No valid trajectories were evaluated.")
    logging.info("Done")


# ===================== PI05 CHANGE: model/data adapter helpers =================
# Main GR00T structure is above. These replace GR00T-specific dependencies.

class Pi05EpisodeLoader:
    """Replace GR00T LeRobotEpisodeLoader using local LeRobot/OpenPI transforms."""

    def __init__(self, path, Dataset, cfg, dc, repack, forward, backward, prompt):
        info = json.loads((path / "meta/info.json").read_text())
        fps = float(info["fps"])
        if fps <= 0:
            raise ValueError(f"Invalid FPS in {path}")
        self.action_keys = tuple(dc.action_sequence_keys)
        missing = set(self.action_keys) - set(info["features"])
        if not self.action_keys or missing:
            raise ValueError(f"Config action keys {self.action_keys}: missing {missing} in {path}")
        # PI05 CHANGE: identical training chunk timestamps [0, 1/fps, ...].
        # Absolute local path belongs in root, never in repo_id.
        self.dataset = Dataset(
            repo_id=path.name, root=path,
            delta_timestamps={k: [i / fps for i in range(cfg.model.action_horizon)]
                              for k in self.action_keys},
            download_videos=False,
        )
        self.spans = episode_spans(self.dataset.hf_dataset["episode_index"])
        self.repack, self.forward, self.backward = repack, forward, backward
        self.prompt = prompt

    def __len__(self):
        return len(self.spans)

    def __getitem__(self, traj_id):
        start, end = self.spans[traj_id]
        return Pi05Trajectory(start, end)


@dataclass
class Pi05Trajectory:
    start: int
    end: int

    def __len__(self):
        return self.end - self.start


def numpy_tree(value):
    # Torch image tensors from LeRobot are CPU tensors. Preserve nested dictionaries.
    if isinstance(value, dict):
        return {k: numpy_tree(v) for k, v in value.items()}
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def discover_datasets(root, names=None):
    root = root.expanduser().resolve()
    if (root / 'meta/info.json').is_file():
        if names:
            raise ValueError('--repo-ids only applies to a parent directory.')
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(p for p in root.iterdir() if p.is_dir() and (p / 'meta/info.json').is_file())
    if names:
        available = {p.name: p for p in paths}
        if set(names) - set(available):
            raise ValueError(f'Unknown children: {set(names) - set(available)}; available: {list(available)}')
        paths = [available[n] for n in dict.fromkeys(names)]
    if not paths:
        raise ValueError(f'No LeRobot datasets found in {root}')
    return paths


def episode_spans(episode_ids):
    ids = np.asarray(episode_ids).reshape(-1).astype(np.int64)
    if len(ids) == 0:
        raise ValueError('Empty dataset')
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(ids)]
    spans = {}
    for start, end in zip(starts, ends):
        eid = int(ids[start])
        if eid in spans:
            raise ValueError(f'Episode {eid} is not contiguous in the dataset.')
        spans[eid] = (int(start), int(end))
    return spans


def metrics(gt, pred):
    if gt.ndim != 2 or gt.shape != pred.shape or not gt.size:
        raise ValueError(f'Invalid action shapes: GT={gt.shape}, predicted={pred.shape}')
    if not np.isfinite(gt).all() or not np.isfinite(pred).all():
        raise ValueError('Nonfinite ground truth/prediction; refusing misleading metrics.')
    error = pred.astype(np.float64) - gt.astype(np.float64)
    return {'mse': float(np.mean(error ** 2)), 'mae': float(np.mean(np.abs(error))),
            'mse_per_dim': np.mean(error ** 2, axis=0).tolist(),
            'mae_per_dim': np.mean(np.abs(error), axis=0).tolist()}


def install_offline_guard():
    # Apply BEFORE importing model/config/tokenizer modules. No source edits required.
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[key] = '1'
    from openpi.shared import download

    def cached_only(url, **kwargs):
        parsed = urlparse(str(url))
        if not parsed.scheme:
            path = Path(url).expanduser().resolve()
        else:
            cache = Path(os.getenv('OPENPI_DATA_HOME', '~/.cache/openpi')).expanduser().resolve()
            path = (cache / parsed.netloc / parsed.path.strip('/')).resolve()
            if not path.is_relative_to(cache):
                raise ValueError('Unsafe cache path')
        if not path.exists():
            raise FileNotFoundError(f'Offline: required local/cached asset is missing: {path}\nOriginal: {url}')
        return path

    download.maybe_download = cached_only


def load_runtime(args):
    # PI05 CHANGE: this replaces Gr00tPolicy/embodiment setup. The local config
    # MUST exist in your repo; a downloaded upstream OpenPI lacks the custom 60-D
    # model/config. No architecture is inferred merely from a checkpoint name.
    # Allow execution as scripts/foo.py from the OpenPI root without pip reinstall.
    local_src = Path.cwd() / 'src'
    if (local_src / 'openpi').is_dir():
        sys.path.insert(0, str(local_src))
    if args.offline:
        install_offline_guard()
    from openpi.training import config
    from openpi import transforms
    from openpi.shared import normalize
    from openpi.policies import policy_config
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as exc:
        if exc.name not in ('lerobot.common', 'lerobot.common.datasets', 'lerobot.common.datasets.lerobot_dataset'):
            raise
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    checkpoint = Path(args.model_path).expanduser().resolve()
    if checkpoint.name == 'params':
        checkpoint = checkpoint.parent
    if not (checkpoint / 'params').is_dir():
        raise ValueError(f'{checkpoint} must contain params/ from a complete OpenPI JAX training checkpoint.')
    cfg = config.get_config(args.config_name)
    # Suppress unrelated training assets/GCS reads during config construction.
    # Model transforms/tokenizer still use the same config and cached tokenizer.
    factory = cfg.data
    if dataclasses.is_dataclass(factory) and hasattr(factory, 'assets'):
        factory = dataclasses.replace(factory, assets=dataclasses.replace(factory.assets, assets_dir=str(checkpoint / 'assets')))
        cfg = dataclasses.replace(cfg, data=factory)
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    horizon = int(cfg.model.action_horizon)
    if args.execution_horizon > horizon:
        raise ValueError(f'--execution-horizon must be <= model action_horizon={horizon}')
    if args.norm_stats_path:
        stats_path = Path(args.norm_stats_path).expanduser().resolve()
        if stats_path.is_dir():
            stats_path = stats_path / 'norm_stats.json'
    else:
        if data_cfg.asset_id is None:
            raise ValueError('Config asset_id is missing; provide --norm-stats-path.')
        stats_path = checkpoint / 'assets' / data_cfg.asset_id / 'norm_stats.json'
    if not stats_path.is_file():
        raise FileNotFoundError(f'Training norm stats missing: {stats_path}; provide --norm-stats-path explicitly.')
    # normalize.load reads norm_stats.json, not an arbitrary filename.
    if stats_path.name != 'norm_stats.json':
        raise ValueError('--norm-stats-path file must be named norm_stats.json')
    stats = normalize.load(stats_path.parent)
    repack = transforms.compose(data_cfg.repack_transforms.inputs)
    forward = transforms.compose(data_cfg.data_transforms.inputs)
    backward = transforms.compose(data_cfg.data_transforms.outputs)
    policy = None
    if not args.dry_run:
        policy = policy_config.create_trained_policy(
            cfg, checkpoint, norm_stats=stats,
            sample_kwargs={'num_steps': args.denoising_steps},
        )
    return cfg, data_cfg, LeRobotDataset, repack, forward, backward, policy, stats_path, checkpoint


def prepare_sample(dataset, index, repack, args):
    # PI05 CHANGE: apply the TRAINING repack once. create_trained_policy applies
    # data/model transforms internally, so applying them here too would double
    # normalize or subtract state twice. Keep GT before those transforms.
    raw = numpy_tree(dataset[index])
    # Tasks are local to each child, avoiding MultiLeRobotDataset task-index collisions.
    if args.prompt is not None:
        raw['prompt'] = args.prompt
    elif 'prompt' not in raw:
        if 'task' in raw and isinstance(raw['task'], str):
            raw['prompt'] = raw['task']
        else:
            from openpi.transforms import PromptFromLeRobotTask
            raw = PromptFromLeRobotTask(dataset.meta.tasks)(raw)
    packed = repack(copy.deepcopy(raw))
    # A repack config may not retain the prompt. Preserve the task chosen above.
    packed.setdefault('prompt', raw['prompt'])
    if 'actions' not in packed or 'state' not in packed:
        raise KeyError(f'Repack must produce state and actions. Actual keys: {list(packed)}')
    gt = np.asarray(packed['actions'], dtype=np.float64)
    if gt.ndim != 2:
        raise ValueError(f'Expected (H,D) ground truth, got {gt.shape}')
    return packed, gt


def check_roundtrip(packed, forward, backward):
    # Ensure policy output transforms return to the same space as repacked raw GT.
    # No normalization here: only action-space input/output conversion is tested.
    mid = forward(copy.deepcopy(packed))
    restored = backward({'state': copy.deepcopy(mid['state']), 'actions': copy.deepcopy(mid['actions'])})
    gt = np.asarray(packed['actions'])
    recovered = np.asarray(restored['actions'])
    if recovered.shape != gt.shape or not np.allclose(gt, recovered, rtol=1e-5, atol=1e-6):
        raise ValueError('Action transforms do not round-trip to raw GT. Check masking, delta/absolute or action ordering.')


def _evaluate_episode_chunks(dataset, start, end, repack, forward, backward, policy, args):
    # PI05 CHANGE: adapter for the original evaluate_single_trajectory loop.
    # LeRobot supplies H future raw actions at each observation using FPS offsets.
    # Model inference still uses ONLY the recorded observation, never the GT chunk.
    count = end - start if args.steps == 0 else min(args.steps, end - start)
    truth, predicted, states, inference_frames = [], [], [], []
    for offset in range(0, count, args.execution_horizon):
        packed, gt_chunk = prepare_sample(dataset, start + offset, repack, args)
        if offset == 0:
            check_roundtrip(packed, forward, backward)
        # Never send ground truth actions to the inference pipeline.
        obs = copy.deepcopy(packed)
        obs.pop('actions')
        output = policy.infer(obs)
        pred_chunk = np.asarray(output['actions'], dtype=np.float64)
        take = min(args.execution_horizon, count - offset)
        if pred_chunk.ndim != 2 or pred_chunk.shape[0] < take or gt_chunk.shape[0] < take:
            raise ValueError(f'Invalid chunk length: predicted={pred_chunk.shape}, GT={gt_chunk.shape}, needed={take}')
        # Do not silently slice excess action dimensions: wrong architecture must fail.
        if pred_chunk.shape[1] != gt_chunk.shape[1]:
            raise ValueError(f'Action dimension mismatch: predicted={pred_chunk.shape}, GT={gt_chunk.shape}')
        truth.append(gt_chunk[:take].copy())
        predicted.append(pred_chunk[:take].copy())
        inference_frames.append(offset)
        # Record actual per-frame states, not a repeated state from the chunk start.
        for j in range(take):
            state_sample, _ = prepare_sample(dataset, start + offset + j, repack, args)
            states.append(np.asarray(state_sample['state']).copy())
        logging.info('Frames %d/%d', offset + take, count)
    gt, pred = np.concatenate(truth), np.concatenate(predicted)
    return gt, pred, np.stack(states), np.asarray(inference_frames), metrics(gt, pred)


def self_test():
    """CPU-only regression checks, independent of OpenPI/LeRobot/GPU installation."""
    from types import SimpleNamespace
    import tempfile
    assert episode_spans([0, 0, 2, 2, 2]) == {0: (0, 2), 2: (2, 5)}
    try:
        episode_spans([0, 1, 0])
        raise AssertionError('noncontiguous episode accepted')
    except ValueError:
        pass
    assert metrics(np.zeros((2, 3)), np.ones((2, 3)))['mse'] == 1
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ('a', 'b'):
            (root / name / 'meta').mkdir(parents=True)
            (root / name / 'meta/info.json').write_text('{}')
        assert [p.name for p in discover_datasets(root)] == ['a', 'b']
        assert [p.name for p in discover_datasets(root, ['b'])] == ['b']
    class FakeDataset:
        def __getitem__(self, index):
            return {'state': np.array([index, index]), 'prompt': 'test',
                    'actions': np.tile(np.arange(index, index + 4)[:, None], (1, 2))}
    class FakePolicy:
        def infer(self, obs):
            assert 'actions' not in obs, 'ground-truth leakage'
            i = obs['state'][0]
            return {'actions': np.tile(np.arange(i, i + 4)[:, None], (1, 2))}
    identity = lambda x: x
    args = SimpleNamespace(steps=0, execution_horizon=3, prompt=None)
    gt, pred, states, marks, result = _evaluate_episode_chunks(FakeDataset(), 0, 5, identity, identity, identity, FakePolicy(), args)
    assert gt.shape == (5, 2) and np.array_equal(gt, pred) and result['mse'] == 0
    assert marks.tolist() == [0, 3] and states[:, 0].tolist() == list(range(5))
    # Absolute->delta->absolute conversion must use the current observation state.
    def delta(x):
        x['actions'] = x['actions'] - x['state'][None, :]
        return x
    def absolute(x):
        x['actions'] = x['actions'] + x['state'][None, :]
        return x
    check_roundtrip(FakeDataset()[2], delta, absolute)
    print('SELF TEST PASSED: metrics, local dataset discovery, episode boundaries, tail clipping, no GT leakage, action-space round trip')

if __name__ == "__main__":
    # PI05 CHANGE: standalone self-test needs no tyro/model dependencies.
    # Normal CLI still uses tyro.cli(ArgsConfig), as in GR00T.
    if sys.argv[1:] == ["--self-test"]:
        main(ArgsConfig(self_test=True))
    else:
        import tyro
        main(tyro.cli(ArgsConfig))

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _require_analysis_deps() -> Tuple[Any, Any, Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless / file output
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
    except Exception as exc:
        print(
            "Missing analysis dependencies. Install with:\n"
            "  python -m pip install matplotlib pandas seaborn numpy\n\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return plt, np, pd, sns


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_home_pose(server_py: Path) -> Tuple[int, int]:
    default_pan, default_tilt = 80, 120
    try:
        text = server_py.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return default_pan, default_tilt

    pan = None
    tilt = None
    m = re.search(r"^HOME_PAN\\s*=\\s*(\\d+)\\s*$", text, flags=re.MULTILINE)
    if m:
        pan = int(m.group(1))
    m = re.search(r"^HOME_TILT\\s*=\\s*(\\d+)\\s*$", text, flags=re.MULTILINE)
    if m:
        tilt = int(m.group(1))

    return (pan if pan is not None else default_pan), (tilt if tilt is not None else default_tilt)


def _need_sign(delta: float, tol: int) -> int:
    if delta > tol:
        return 1
    if delta < -tol:
        return -1
    return 0


_DIRECTIONAL_GESTURE_MAP: Dict[str, Tuple[str, int]] = {
    "ArrowLeft": ("pan", -1),
    "ArrowRight": ("pan", 1),
    "ArrowUp": ("tilt", 1),
    "ArrowDown": ("tilt", -1),
}


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "":
            return None
        try:
            return int(cleaned)
        except ValueError:
            try:
                return int(float(cleaned))
            except ValueError:
                return None
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned: List[float] = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return float(sum(cleaned) / len(cleaned))


def _parse_iso_ts(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    try:
        # validate only; keep original string so json stays identical across locales
        datetime.fromisoformat(value)
        return value
    except ValueError:
        return value


@dataclass(frozen=True)
class BlockQuestionnaire:
    comfort: Optional[int]
    performance: Optional[int]
    diagnostic: Optional[str]

    in_control: Optional[int]
    expected: Optional[int]
    confident: Optional[int]
    recover: Optional[int]

    self_conscious: Optional[int]
    judged: Optional[int]
    held_back: Optional[int]

    justified_social: Optional[int]
    justified_bystanders: Optional[int]
    justified_predict: Optional[int]

    mental: Optional[int]
    physical: Optional[int]
    effort: Optional[int]
    frustration: Optional[int]

    @property
    def control_confidence_mean(self) -> Optional[float]:
        return _mean([self.in_control, self.expected, self.confident, self.recover])

    @property
    def social_mean(self) -> Optional[float]:
        return _mean([self.self_conscious, self.judged, self.held_back])

    @property
    def justification_mean(self) -> Optional[float]:
        return _mean([self.justified_social, self.justified_bystanders, self.justified_predict])

    @property
    def workload_mean(self) -> Optional[float]:
        return _mean([self.mental, self.physical, self.effort, self.frustration])


def _parse_block_questionnaire(payload: Dict[str, Any]) -> BlockQuestionnaire:
    control = payload.get("control_confidence") or {}
    social = payload.get("social") or {}
    justify = payload.get("justification") or {}
    workload = payload.get("workload") or {}

    return BlockQuestionnaire(
        comfort=_to_int(payload.get("comfort")),
        performance=_to_int(payload.get("performance")),
        diagnostic=payload.get("diagnostic"),
        in_control=_to_int(control.get("in_control")),
        expected=_to_int(control.get("expected")),
        confident=_to_int(control.get("confident")),
        recover=_to_int(control.get("recover")),
        self_conscious=_to_int(social.get("self_conscious")),
        judged=_to_int(social.get("judged")),
        held_back=_to_int(social.get("held_back")),
        justified_social=_to_int(justify.get("social")),
        justified_bystanders=_to_int(justify.get("bystanders")),
        justified_predict=_to_int(justify.get("predict")),
        mental=_to_int(workload.get("mental")),
        physical=_to_int(workload.get("physical")),
        effort=_to_int(workload.get("effort")),
        frustration=_to_int(workload.get("frustration")),
    )


def _safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    if n is None or d is None:
        return None
    if d == 0:
        return None
    return float(n / d)


def _describe_numeric(series) -> Dict[str, Any]:
    # series is expected to be a pandas Series (but keep it library-agnostic in tests)
    values = [float(v) for v in series.dropna().tolist()]  # type: ignore[attr-defined]
    if not values:
        return {"n": 0}
    values_sorted = sorted(values)
    n = len(values_sorted)
    mean_v = sum(values_sorted) / n
    median_v = values_sorted[n // 2] if n % 2 == 1 else (values_sorted[n // 2 - 1] + values_sorted[n // 2]) / 2
    if n >= 2:
        var = sum((v - mean_v) ** 2 for v in values_sorted) / (n - 1)
        sd = math.sqrt(var)
    else:
        sd = None
    return {
        "n": n,
        "mean": mean_v,
        "median": median_v,
        "sd": sd,
        "min": values_sorted[0],
        "max": values_sorted[-1],
    }


def _paired_stats(np, x: List[float], y: List[float], seed: int = 42) -> Dict[str, Any]:
    """
    Returns paired differences stats for y - x.
    Uses bootstrap CI (percentile) for robustness and minimal assumptions.
    """
    if len(x) != len(y):
        raise ValueError("paired_stats inputs must have same length")
    if not x:
        return {"n": 0}
    diffs = np.array([yy - xx for xx, yy in zip(x, y)], dtype=float)
    n = int(diffs.size)
    mean_diff = float(diffs.mean())
    sd_diff = float(diffs.std(ddof=1)) if n >= 2 else None
    d_z = float(mean_diff / sd_diff) if (sd_diff not in (None, 0.0)) else None

    rng = np.random.default_rng(seed)
    n_boot = 20000 if n >= 3 else 5000
    boot_means = rng.choice(diffs, size=(n_boot, n), replace=True).mean(axis=1)
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975]).tolist()
    return {
        "n": n,
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "cohen_dz": d_z,
        "ci95_mean_diff_bootstrap": [float(ci_low), float(ci_high)],
    }


def _infer_block_key(preset: str, environment: str) -> str:
    env = environment.lower()
    if "public" in env:
        env_key = "public"
    elif "private" in env:
        env_key = "private"
    else:
        env_key = "unknown"
    return f"{preset}_{env_key}"



def _normalize_environment(environment: Optional[str]) -> str:
    """Map free-form environment labels to canonical keys used in analysis."""
    env = (environment or "").strip().lower()
    if "public" in env:
        return "public"
    if "private" in env:
        return "private"
    return (environment or "").strip() or "unknown"


def _extract_gesture_counts(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    Returns:
      {kind: {name: count}}
    """
    out: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in events:
        if e.get("type") != "gesture":
            continue
        kind = str(e.get("kind") or "unknown")
        name = str(e.get("name") or "unknown")
        out[kind][name] += 1
    return {k: dict(v) for k, v in out.items()}


def _flatten_gesture_counts(kind_to_counts: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for kind, counts in kind_to_counts.items():
        for name, count in counts.items():
            rows.append({"gesture_kind": kind, "gesture_name": name, "count": int(count)})
    rows.sort(key=lambda r: (r["gesture_kind"], r["gesture_name"]))
    return rows


def _plot_save(fig, out_dir: Path, stem: str) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    return [str(png_path), str(pdf_path)]


def _make_plots(plt, np, pd, sns, blocks_df, targets_df, final_df, target_direction_df, out_dir: Path) -> List[Dict[str, Any]]:
    sns.set_theme(style="whitegrid", context="paper", palette="colorblind")

    plots: List[Dict[str, Any]] = []
    plots_dir = out_dir / "plots"

    env_label_map: Dict[str, str] = {}
    try:
        if (blocks_df is not None) and (not blocks_df.empty) and {"environment_key", "environment"}.issubset(set(blocks_df.columns)):
            for key, g in blocks_df.dropna(subset=["environment_key"]).groupby("environment_key"):
                values = g["environment"].dropna().astype(str)
                if not values.empty:
                    env_label_map[str(key)] = values.value_counts().idxmax()
    except Exception:
        env_label_map = {}

    def env_label(key: Any) -> str:
        if key is None:
            return "Unknown"
        try:
            if isinstance(key, float) and math.isnan(key):
                return "Unknown"
        except Exception:
            return "Unknown"
        return env_label_map.get(str(key), str(key).capitalize())

    def add_plot(stem: str, description: str, files: List[str], notes: Optional[str] = None) -> None:
        plots.append({"stem": stem, "description": description, "files": files, "notes": notes})

    # Objective: block-level time and gesture totals by preset
    for metric, ylabel, stem in [
        ("elapsed_total_s", "Total block time (s)", "elapsed_total_by_preset"),
        ("gestures_total", "Total gestures (block)", "gestures_total_by_preset"),
    ]:
        df = blocks_df.dropna(subset=[metric]).copy()
        if df.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        if "environment_key" not in df.columns and "environment" in df.columns:
            df["environment_key"] = df["environment"].map(_normalize_environment)
        df["environment_label"] = df["environment_key"].map(env_label) if "environment_key" in df.columns else "Unknown"
        sns.boxplot(data=df, x="preset", y=metric, hue="environment_label", ax=ax, showfliers=False)
        ax.set_xlabel("Gesture set")
        ax.set_ylabel(ylabel)
        ax.legend(title="Context", loc="best")
        files = _plot_save(fig, plots_dir, stem)
        plt.close(fig)
        add_plot(stem, f"Median/IQR of {ylabel} by gesture set.", files)

    # Objective: cumulative progress to reach target N (N=1..5),
    # showing completed blocks vs early stops on the attempted target.
    if not targets_df.empty:
        progress = targets_df.copy()
        if "environment_key" not in progress.columns:
            progress["environment_key"] = progress["environment"].map(_normalize_environment)
        progress = progress.dropna(subset=["target_index"]).copy()
        progress["target_n"] = progress["target_index"].astype(int) + 1
        progress["status"] = np.where(progress["completed"] == True, "completed", "stopped")  # noqa: E712

        group_cols = ["session_id", "block_key", "participant_id", "preset", "environment_key"]
        for col in ["seconds", "gestures"]:
            if col in progress.columns:
                progress[col] = pd.to_numeric(progress[col], errors="coerce")
        progress = progress.sort_values(group_cols + ["target_n"])
        if "seconds" in progress.columns:
            progress["seconds_cum"] = progress.groupby(group_cols)["seconds"].cumsum()
        if "gestures" in progress.columns:
            progress["gestures_cum"] = progress.groupby(group_cols)["gestures"].cumsum()

        def make_progress_plot(metric: str, ylabel: str, stem: str) -> None:
            from matplotlib.patches import Patch

            df = progress.dropna(subset=[metric]).copy()
            if df.empty:
                return

            env_order = [k for k in ["private", "public", "unknown"] if k in set(df["environment_key"].dropna())]
            if not env_order:
                env_order = sorted(set(df["environment_key"].dropna().tolist()))

            preset_order = [p for p in ["discreet", "expressive"] if p in set(df["preset"].dropna())]
            if not preset_order:
                preset_order = sorted(set(df["preset"].dropna().tolist()))

            status_order = ["completed", "stopped"]
            palette = sns.color_palette("colorblind", n_colors=max(2, len(preset_order)))
            preset_color = {p: palette[i % len(palette)] for i, p in enumerate(preset_order)}
            preset_label = {"discreet": "Discreet", "expressive": "Expressive"}
            status_label = {"completed": "Completed", "stopped": "Stopped (partial)"}

            fig, axes = plt.subplots(
                nrows=1,
                ncols=max(1, len(env_order)),
                figsize=(6.5, 3.6),
                sharey=True,
                squeeze=False,
            )

            target_ns = [1, 2, 3, 4, 5]
            group_width = 0.86
            preset_width = group_width / max(1, len(preset_order))
            status_width = preset_width / max(1, len(status_order))

            y_top_candidate = 0.0
            for col_idx, env_key in enumerate(env_order):
                ax = axes[0][col_idx]
                env_df = df[df["environment_key"] == env_key].copy()
                if env_df.empty:
                    ax.axis("off")
                    continue
                y_pad = 0.0
                try:
                    y_max = float(env_df[metric].max())
                    y_pad = 0.018 * y_max if y_max > 0 else 0.0
                except Exception:
                    y_pad = 0.0

                grouped = env_df.groupby(["preset", "status", "target_n"])[metric]
                stats = grouped.agg(n="count", median="median").reset_index()
                quant = grouped.quantile([0.25, 0.75]).unstack(level=-1).reset_index()
                quant = quant.rename(columns={0.25: "q1", 0.75: "q3"})
                stats = stats.merge(quant, on=["preset", "status", "target_n"], how="left")
                try:
                    q3_max = float(stats["q3"].max()) if stats["q3"].notna().any() else float(stats["median"].max())
                    y_top_candidate = max(y_top_candidate, q3_max + (4.0 * y_pad))
                except Exception:
                    pass

                for t in target_ns:
                    for p_i, preset in enumerate(preset_order):
                        for s_i, status in enumerate(status_order):
                            row = stats[
                                (stats["target_n"] == t) & (stats["preset"] == preset) & (stats["status"] == status)
                            ]
                            if row.empty:
                                continue
                            r = row.iloc[0]
                            x = (
                                float(t)
                                - group_width / 2
                                + p_i * preset_width
                                + s_i * status_width
                                + status_width / 2
                            )
                            y = float(r["median"])
                            q1 = float(r.get("q1")) if r.get("q1") is not None else y
                            q3 = float(r.get("q3")) if r.get("q3") is not None else y
                            n = int(r["n"])

                            hatch = "///" if status == "stopped" else None
                            alpha = 0.55 if status == "stopped" else 1.0
                            bar = ax.bar(
                                [x],
                                [y],
                                width=status_width * 0.92,
                                color=preset_color.get(preset),
                                edgecolor="black",
                                linewidth=0.6,
                                hatch=hatch,
                                alpha=alpha,
                                zorder=2,
                            )
                            _ = bar  # silence linter in minimal environments

                            err_low = max(0.0, y - q1)
                            err_high = max(0.0, q3 - y)
                            if err_low > 0 or err_high > 0:
                                ax.errorbar(
                                    [x],
                                    [y],
                                    yerr=[[err_low], [err_high]],
                                    fmt="none",
                                    ecolor="black",
                                    elinewidth=0.8,
                                    capsize=2.5,
                                    zorder=3,
                                )
                            ax.text(
                                x,
                                y + y_pad,
                                f"n={n}",
                                ha="center",
                                va="bottom",
                                fontsize=7,
                                color="black",
                                alpha=0.9,
                                zorder=4,
                            )

                ax.set_title(env_key.capitalize())
                ax.set_xlabel("Targets reached (1-5)")
                ax.set_xticks(target_ns)
                ax.set_xlim(0.5, 5.5)
                ax.set_ylim(bottom=0)
                if col_idx == 0:
                    ax.set_ylabel(ylabel)
                else:
                    ax.set_ylabel("")

                if col_idx == 0:
                    handles = []
                    labels = []
                    for preset in preset_order:
                        for status in status_order:
                            handles.append(
                                Patch(
                                    facecolor=preset_color.get(preset),
                                    edgecolor="black",
                                    hatch="///" if status == "stopped" else None,
                                    alpha=0.55 if status == "stopped" else 1.0,
                                )
                            )
                            labels.append(f"{preset_label.get(preset, preset)} - {status_label.get(status, status)}")
                    ax.legend(handles, labels, fontsize=7, frameon=True, loc="upper left")

            if y_top_candidate > 0:
                y_top = math.ceil(y_top_candidate / 10.0) * 10.0
                for ax in axes[0]:
                    if ax.has_data():
                        ax.set_ylim(0, y_top)

            files = _plot_save(fig, plots_dir, stem)
            plt.close(fig)
            add_plot(
                stem,
                f"Median/IQR of {ylabel} to reach target N (N=1-5), split by gesture set; hatched bars indicate early-stopped blocks on the attempted target.",
                files,
                notes="Bars use per-block cumulative progress; sample size (n) is printed on each bar. Hatched bars are partial (block stopped before completing that target).",
            )

        if "seconds_cum" in progress.columns:
            make_progress_plot("seconds_cum", "Cumulative time (s)", "time_to_targets_progress_by_preset")
        if "gestures_cum" in progress.columns:
            make_progress_plot("gestures_cum", "Cumulative gestures", "gestures_to_targets_progress_by_preset")

    # Objective: completion counts and rates
    if "targets_completed" in blocks_df.columns:
        df = blocks_df.dropna(subset=["targets_completed"]).copy()
        if not df.empty:
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            if "environment_key" not in df.columns and "environment" in df.columns:
                df["environment_key"] = df["environment"].map(_normalize_environment)
            df["environment_label"] = df["environment_key"].map(env_label) if "environment_key" in df.columns else "Unknown"
            sns.boxplot(data=df, x="preset", y="targets_completed", hue="environment_label", ax=ax, showfliers=False)
            ax.set_xlabel("Gesture set")
            ax.set_ylabel("Completed targets (out of 5)")
            ax.legend(title="Context", loc="best")
            ax.set_ylim(-0.25, 5.25)
            files = _plot_save(fig, plots_dir, "targets_completed_by_preset")
            plt.close(fig)
            add_plot(
                "targets_completed_by_preset",
                "Median/IQR of completed targets (0–5) by gesture set.",
                files,
            )

    if "completion_rate" in blocks_df.columns:
        df = blocks_df.dropna(subset=["completion_rate"]).copy()
        if not df.empty:
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            if "environment_key" not in df.columns and "environment" in df.columns:
                df["environment_key"] = df["environment"].map(_normalize_environment)
            df["environment_label"] = df["environment_key"].map(env_label) if "environment_key" in df.columns else "Unknown"
            sns.boxplot(data=df, x="preset", y="completion_rate", hue="environment_label", ax=ax, showfliers=False)
            ax.set_xlabel("Gesture set")
            ax.set_ylabel("Completion rate")
            ax.legend(title="Context", loc="best")
            ax.set_ylim(-0.05, 1.05)
            files = _plot_save(fig, plots_dir, "completion_rate_by_preset")
            plt.close(fig)
            add_plot(
                "completion_rate_by_preset",
                "Median/IQR of completion rate by gesture set.",
                files,
            )

    # Per-target time/gestures
    if not targets_df.empty:
        for metric, ylabel, stem in [
            ("seconds", "Time per target (s)", "time_per_target_by_preset"),
            ("gestures", "Gestures per target", "gestures_per_target_by_preset"),
        ]:
            df = targets_df[targets_df["completed"] == True].copy()
            if df.empty:
                continue
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            sns.lineplot(
                data=df,
                x="target_index",
                y=metric,
                hue="preset",
                ax=ax,
                marker="o",
                errorbar=("ci", 95),
            )
            ax.set_xlabel("Target index")
            ax.set_ylabel(ylabel)
            files = _plot_save(fig, plots_dir, stem)
            plt.close(fig)
            add_plot(
                stem,
                f"Mean {ylabel} by target index and gesture set (line=mean; shaded=95% CI).",
                files,
                notes="95% CI computed by seaborn.",
            )

    # Per-target gesture counts (distribution)
    if not targets_df.empty and "gestures" in targets_df.columns:
        df = targets_df[targets_df["completed"] == True].dropna(subset=["gestures"]).copy()  # noqa: E712
        if not df.empty:
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            sns.boxplot(data=df, x="target_index", y="gestures", hue="preset", ax=ax, showfliers=False)
            ax.set_xlabel("Target index")
            ax.set_ylabel("Gestures to reach target")
            files = _plot_save(fig, plots_dir, "gestures_per_target_distribution")
            plt.close(fig)
            add_plot(
                "gestures_per_target_distribution",
                "Median/IQR of gestures required per target index, split by gesture set.",
                files,
            )

    # Expressive directional accuracy (proxy for miss inputs)
    if target_direction_df is not None and not target_direction_df.empty:
        df = target_direction_df.copy()
        if "preset" in df.columns:
            df = df[df["preset"] == "expressive"].copy()
        needed_cols = {"gestures_towards", "gestures_away", "gestures_off_axis", "target_index"}
        if not df.empty and needed_cols.issubset(set(df.columns)):
            summed = (
                df.groupby("target_index")[["gestures_towards", "gestures_away", "gestures_off_axis"]]
                .sum(numeric_only=True)
                .reset_index()
                .sort_values("target_index")
            )
            totals = (
                summed["gestures_towards"].astype(float)
                + summed["gestures_away"].astype(float)
                + summed["gestures_off_axis"].astype(float)
            )
            summed = summed[totals > 0].copy()
            totals = totals[totals > 0]
            if not summed.empty:
                pct_towards = 100.0 * summed["gestures_towards"].astype(float) / totals
                pct_away = 100.0 * summed["gestures_away"].astype(float) / totals
                pct_off = 100.0 * summed["gestures_off_axis"].astype(float) / totals

                fig, ax = plt.subplots(figsize=(6.5, 3.6))
                x = summed["target_index"].tolist()
                bottom = np.zeros(len(summed), dtype=float)
                ax.bar(x, pct_towards, bottom=bottom, label="Towards target", color="#009E73")
                bottom = bottom + pct_towards.to_numpy()
                ax.bar(x, pct_away, bottom=bottom, label="Away from target", color="#D55E00")
                bottom = bottom + pct_away.to_numpy()
                ax.bar(x, pct_off, bottom=bottom, label="Off-axis", color="#999999")

                ax.set_xlabel("Target index")
                ax.set_ylabel("Gestures (%)")
                ax.set_ylim(0, 100)
                ax.set_xticks(x)
                ax.legend(loc="upper right", frameon=True)

                files = _plot_save(fig, plots_dir, "expressive_direction_share_by_target")
                plt.close(fig)
                add_plot(
                    "expressive_direction_share_by_target",
                    "Expressive only: share of Arrow-key gestures classified as towards/away/off-axis vs the start-of-target delta (HOME→target0; target[i-1]→target[i]).",
                    files,
                    notes="Classification uses only Arrow key gesture events; it does not model overshoot/backtracking within a target.",
                )

    # Questionnaire composites by preset
    composite_cols = [
        ("comfort", "Comfort (1–6)"),
        ("performance", "Self-rated performance (1–7)"),
        ("control_confidence_mean", "Control+confidence mean (1–7)"),
        ("social_mean", "Self-consciousness mean (1–7)"),
        ("justification_mean", "Justification mean (1–7)"),
        ("workload_mean", "Workload mean (1–7)"),
    ]
    available = [c for c, _ in composite_cols if c in blocks_df.columns and not blocks_df[c].dropna().empty]
    if available:
        n = len(available)
        ncols = 3
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5, 2.2 * nrows), squeeze=False)
        for i, (col, label) in enumerate([item for item in composite_cols if item[0] in available]):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            df = blocks_df.dropna(subset=[col]).copy()
            sns.boxplot(data=df, x="preset", y=col, ax=ax, showfliers=False)
            ax.set_xlabel("")
            ax.set_ylabel(label)
        for j in range(n, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")
        files = _plot_save(fig, plots_dir, "questionnaire_ratings_grid")
        plt.close(fig)
        add_plot(
            "questionnaire_ratings_grid",
            "Median/IQR of block questionnaire ratings by gesture set (comfort, performance, control/confidence, social, justification, workload).",
            files,
        )

    # Final comparison (Likert 1=Discreet, 7=Expressive)
    compare_cols = [
        ("prefer_public", "Prefer in public (1=Discreet, 7=Expressive)"),
        ("prefer_private", "Prefer in private (1=Discreet, 7=Expressive)"),
        ("more_precise", "Prefer for precision (1=Discreet, 7=Expressive)"),
        ("more_embarrassing", "Avoid embarrassment (1=Discreet, 7=Expressive)"),
        ("more_justified", "Feel justified (1=Discreet, 7=Expressive)"),
    ]
    compare_available = [c for c, _ in compare_cols if c in final_df.columns and not final_df[c].dropna().empty]
    if compare_available:
        df = final_df.copy()
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        melted = df.melt(id_vars=["participant_id"], value_vars=compare_available, var_name="question", value_name="rating")
        label_map = {c: l for c, l in compare_cols}
        melted["question_label"] = melted["question"].map(label_map).fillna(melted["question"])
        sns.pointplot(
            data=melted,
            x="rating",
            y="question_label",
            ax=ax,
            linestyles="none",
            errorbar=("ci", 95),
            capsize=0.15,
        )
        ax.set_xlim(0.5, 7.5)
        ax.set_xlabel("Rating")
        ax.set_ylabel("")
        files = _plot_save(fig, plots_dir, "final_comparison_ratings")
        plt.close(fig)
        add_plot(
            "final_comparison_ratings",
            "Final comparison ratings (1=Discreet, 7=Expressive): mean with 95% CI.",
            files,
            notes="Point = mean; whiskers = 95% CI.",
        )

    # Acceptability maps: counts per category
    for key, title, stem, order in [
        ("discreet_people", "Discreet: acceptable people", "acceptability_discreet_people", None),
        ("expressive_people", "Expressive: acceptable people", "acceptability_expressive_people", None),
        ("discreet_locations", "Discreet: acceptable locations", "acceptability_discreet_locations", None),
        ("expressive_locations", "Expressive: acceptable locations", "acceptability_expressive_locations", None),
    ]:
        if key not in final_df.columns:
            continue
        counts = Counter()
        responders = 0
        for items in final_df[key].dropna().tolist():
            if isinstance(items, list):
                responders += 1
                counts.update([str(x) for x in items])
        if not counts or responders <= 0:
            continue
        df = pd.DataFrame(
            [{"category": k, "pct": 100.0 * float(v) / float(responders)} for k, v in counts.items()]
        )
        df = df.sort_values("pct", ascending=False)
        if order:
            df["category"] = pd.Categorical(df["category"], categories=order, ordered=True)
            df = df.sort_values("category")
        fig, ax = plt.subplots(figsize=(6.5, max(3.2, 0.28 * len(df))))
        sns.barplot(data=df, x="pct", y="category", ax=ax)
        ax.set_xlabel("Participants selecting (%)")
        ax.set_xlim(0, 100)
        ax.set_ylabel("")
        files = _plot_save(fig, plots_dir, stem)
        plt.close(fig)
        add_plot(
            stem,
            f"Percentage of participants selecting each category: {title}.",
            files,
            notes="Denominator = participants with a response for this item.",
        )

    return plots


@dataclass
class CollectedRows:
    session_dirs: List[Path]
    participants: List[Dict[str, Any]]
    blocks: List[Dict[str, Any]]
    targets: List[Dict[str, Any]]
    gestures: List[Dict[str, Any]]
    final: List[Dict[str, Any]]
    target_direction: List[Dict[str, Any]]
    warnings: List[str]


def _collect_rows(results_dir: Path, home_pan: int, home_tilt: int) -> CollectedRows:
    def rel(path: Path) -> str:
        try:
            return path.relative_to(results_dir).as_posix()
        except Exception:
            return str(path)

    participants_rows: List[Dict[str, Any]] = []
    blocks_rows: List[Dict[str, Any]] = []
    targets_rows: List[Dict[str, Any]] = []
    gesture_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []
    target_direction_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    session_dirs = sorted([p for p in results_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    for session_dir in session_dirs:
        questions_path = session_dir / "questions.json"
        if not questions_path.exists():
            warnings.append(f"[{session_dir.name}] Missing questions.json")
            continue

        questions = _load_json(questions_path)
        # Prefer the directory prefix as the stable participant identifier. The free-text
        # `user_id_text` can be mistyped, which can accidentally merge participants.
        dir_participant_id = str(session_dir.name.split("_")[0]).strip()
        typed_participant_id = str(questions.get("user_id_text") or "").strip()
        participant_id = (
            typed_participant_id
            if typed_participant_id and typed_participant_id.lower() == dir_participant_id.lower()
            else dir_participant_id
        )

        consent = questions.get("consent") or {}
        consent_entries: List[Tuple[str, Dict[str, Any]]] = []
        if isinstance(consent, dict):
            for k, v in consent.items():
                if isinstance(v, dict):
                    consent_entries.append((str(k), v))

        consent_first = consent_entries[0][1] if consent_entries else {}
        condition_order = consent_first.get("condition_order")
        session_environment = consent_first.get("environment")

        prestudy = questions.get("prestudy") or {}
        baseline = questions.get("baseline") or {}
        participants_rows.append(
            {
                "session_id": session_dir.name,
                "participant_id": participant_id,
                "participant_id_typed": typed_participant_id if typed_participant_id else None,
                "condition_order": condition_order,
                "session_environment": session_environment,
                "questions_file": rel(questions_path),
                "consent": consent,
                "age": _to_int(prestudy.get("age")),
                "gender": prestudy.get("gender"),
                "handedness": prestudy.get("handedness"),
                "experience": prestudy.get("experience") if isinstance(prestudy.get("experience"), list) else None,
                "baseline_comfortable": _to_int(baseline.get("comfortable")),
                "baseline_worry": _to_int(baseline.get("worry")),
                "baseline_avoid": _to_int(baseline.get("avoid")),
            }
        )

        blocks = questions.get("blocks") or {}
        if not isinstance(blocks, dict):
            warnings.append(f"[{session_dir.name}] questions.blocks is not an object")
            continue

        for block_key, block_info in blocks.items():
            if not isinstance(block_info, dict):
                continue
            preset = str(block_info.get("preset") or "")
            environment = str(block_info.get("environment") or "")

            answers = block_info.get("answers") if isinstance(block_info.get("answers"), dict) else None
            questionnaire = _parse_block_questionnaire(answers) if answers else None

            run_path = session_dir / f"{block_key}.json"
            if not run_path.exists() and preset and environment:
                run_path = session_dir / f"{_infer_block_key(preset, environment)}.json"
            if not run_path.exists():
                warnings.append(f"[{session_dir.name}] Missing run json for block '{block_key}'")
                continue

            run = _load_json(run_path)

            targets = run.get("targets") if isinstance(run.get("targets"), list) else []
            results = run.get("results") if isinstance(run.get("results"), list) else []
            events = run.get("events") if isinstance(run.get("events"), list) else []
            session_meta = run.get("session") if isinstance(run.get("session"), dict) else {}

            preset_value = run.get("preset") or preset
            environment_label = session_meta.get("environment") or environment
            environment_key = _normalize_environment(environment_label)

            targets_total = int(len(targets))
            current_idx = _to_int(run.get("current_target_index"))
            current_elapsed_s = _to_float(run.get("current_target_elapsed"))

            # Some runs may include a partial record for the currently-attempted target inside
            # `results`. Treat `current_target_index` as the authoritative boundary:
            # - completed: indices < current_target_index
            # - partial (if any): index == current_target_index (handled separately below)
            results_by_index: Dict[int, Dict[str, Any]] = {}
            duplicate_indices: set = set()
            for r in results:
                if not isinstance(r, dict):
                    continue
                idx = _to_int(r.get("index"))
                if idx is None:
                    continue
                if targets_total and not (0 <= idx < targets_total):
                    continue
                if idx in results_by_index:
                    duplicate_indices.add(idx)
                results_by_index[int(idx)] = r

            if duplicate_indices:
                warnings.append(
                    f"[{session_dir.name}] {run_path.name}: Duplicate result indices {sorted(duplicate_indices)}; using last occurrence"
                )

            reason = str(run.get("reason") or "")
            completed_indices = set(results_by_index.keys())
            if reason != "completed" and current_idx is not None:
                completed_indices = {i for i in completed_indices if i < int(current_idx)}
            elif reason == "completed" and targets_total:
                completed_indices = {i for i in completed_indices if i < targets_total}

            if reason == "completed" and targets_total:
                targets_completed = targets_total
            elif current_idx is not None:
                targets_completed = max(0, min(int(current_idx), targets_total)) if targets_total else max(0, int(current_idx))
            else:
                targets_completed = int(len(completed_indices))

            completed_results = [results_by_index[i] for i in sorted(completed_indices)]
            if reason != "completed" and targets_completed and len(completed_results) != int(targets_completed):
                warnings.append(
                    f"[{session_dir.name}] {run_path.name}: current_target_index implies {targets_completed} completed, but results contain {len(completed_results)} completed indices"
                )

            elapsed_total_s = _to_float(run.get("elapsed_total"))
            gestures_total = _to_int(run.get("gestures_total"))

            sum_completed_seconds = sum(_to_float(r.get("seconds")) or 0.0 for r in completed_results)
            sum_completed_gestures = sum(_to_int(r.get("gestures")) or 0 for r in completed_results)

            seconds_per_completed_target_mean = (
                _safe_div(sum_completed_seconds, len(completed_results)) if completed_results else None
            )
            gestures_per_completed_target_mean = (
                _safe_div(sum_completed_gestures, len(completed_results)) if completed_results else None
            )

            time_current_s = None
            if elapsed_total_s is not None:
                time_current_s = max(0.0, float(elapsed_total_s - sum_completed_seconds))
            gestures_current = None
            if gestures_total is not None:
                gestures_current = max(0, int(gestures_total - sum_completed_gestures))

            
            # First-3-target metrics: sum completed targets 0..2 and, if the block ends early,
            # include partial progress on the current target when it falls within the first three.
            time_first3_s = sum(
                _to_float(results_by_index[i].get("seconds")) or 0.0 for i in completed_indices if i < 3
            )
            gestures_first3 = sum(
                _to_int(results_by_index[i].get("gestures")) or 0 for i in completed_indices if i < 3
            )

            if reason != "completed" and current_idx is not None and current_idx < 3:
                add_time = current_elapsed_s if current_elapsed_s is not None else time_current_s
                if add_time is not None:
                    time_first3_s += float(add_time)
                if gestures_current is not None:
                    gestures_first3 += int(gestures_current)

            if questionnaire is None:
                bq = run.get("block_questionnaire")
                if isinstance(bq, dict):
                    questionnaire = _parse_block_questionnaire(bq)

            gesture_events_count = sum(1 for e in events if isinstance(e, dict) and e.get("type") == "gesture")
            target_success_events_count = sum(
                1 for e in events if isinstance(e, dict) and e.get("type") == "target_success"
            )
            if target_success_events_count and targets_completed and target_success_events_count != int(targets_completed):
                warnings.append(
                    f"[{session_dir.name}] {run_path.name}: target_success events ({target_success_events_count}) != inferred targets_completed ({targets_completed})"
                )

            gesture_counts = _extract_gesture_counts([e for e in events if isinstance(e, dict)])
            for row in _flatten_gesture_counts(gesture_counts):
                gesture_rows.append(
                    {
                        "session_id": session_dir.name,
                        "participant_id": participant_id,
                        "block_key": str(block_key),
                        "preset": preset_value,
                        "environment": environment_label,
                        "environment_key": environment_key,
                        **row,
                    }
                )

            tol = _to_int(run.get("tolerance")) or 0
            completed_target_indices = {int(i) for i in completed_indices}

            target_plan: Dict[int, Tuple[int, int]] = {}
            for i, t in enumerate(targets):
                if not isinstance(t, dict):
                    continue
                pan = _to_int(t.get("pan"))
                tilt = _to_int(t.get("tilt"))
                if pan is None or tilt is None:
                    continue
                target_plan[i] = (pan, tilt)

            start_pose: Dict[int, Tuple[int, int]] = {}
            prev_pan, prev_tilt = home_pan, home_tilt
            for i in range(len(targets)):
                start_pose[i] = (prev_pan, prev_tilt)
                if i in target_plan:
                    prev_pan, prev_tilt = target_plan[i]

            direction_counts: Dict[int, Counter[str]] = defaultdict(Counter)
            for e in events:
                if not isinstance(e, dict):
                    continue
                if e.get("type") != "gesture":
                    continue
                if str(e.get("kind") or "") != "key":
                    continue
                name = str(e.get("name") or "")
                if name not in _DIRECTIONAL_GESTURE_MAP:
                    continue
                target_idx = _to_int(e.get("target_index"))
                if target_idx is None:
                    continue
                if target_idx not in target_plan or target_idx not in start_pose:
                    continue
                axis, dir_sign = _DIRECTIONAL_GESTURE_MAP[name]
                start_pan, start_tilt = start_pose[target_idx]
                target_pan, target_tilt = target_plan[target_idx]
                if axis == "pan":
                    need = _need_sign(float(target_pan - start_pan), tol)
                else:
                    need = _need_sign(float(target_tilt - start_tilt), tol)
                if need == 0:
                    cls = "off_axis"
                elif dir_sign == need:
                    cls = "towards"
                else:
                    cls = "away"
                direction_counts[target_idx][cls] += 1

            def sign_to_needed_direction(axis: str, sign: int) -> str:
                if sign == 0:
                    return "none"
                if axis == "pan":
                    return "right" if sign > 0 else "left"
                return "up" if sign > 0 else "down"

            for target_idx, counts in sorted(direction_counts.items()):
                towards = int(counts.get("towards", 0))
                away = int(counts.get("away", 0))
                off_axis = int(counts.get("off_axis", 0))
                total = towards + away + off_axis
                if total <= 0:
                    continue

                start_pan, start_tilt = start_pose[target_idx]
                target_pan, target_tilt = target_plan[target_idx]
                need_pan_sign = _need_sign(float(target_pan - start_pan), tol)
                need_tilt_sign = _need_sign(float(target_tilt - start_tilt), tol)

                target_direction_rows.append(
                    {
                        "session_id": session_dir.name,
                        "participant_id": participant_id,
                        "block_key": str(block_key),
                        "preset": preset_value,
                        "environment": environment_label,
                        "environment_key": environment_key,
                        "tolerance": tol,
                        "target_index": int(target_idx),
                        "target_completed": bool(target_idx in completed_target_indices),
                        "start_pan": int(start_pan),
                        "start_tilt": int(start_tilt),
                        "target_pan": int(target_pan),
                        "target_tilt": int(target_tilt),
                        "need_pan_direction": sign_to_needed_direction("pan", need_pan_sign),
                        "need_tilt_direction": sign_to_needed_direction("tilt", need_tilt_sign),
                        "gestures_total": int(total),
                        "gestures_towards": int(towards),
                        "gestures_away": int(away),
                        "gestures_off_axis": int(off_axis),
                        "pct_towards": float(towards / total),
                        "pct_away": float(away / total),
                        "pct_off_axis": float(off_axis / total),
                        "pct_miss": float((away + off_axis) / total),
                    }
                )

            blocks_rows.append(
                {
                    "session_id": session_dir.name,
                    "participant_id": participant_id,
                    "condition_order": condition_order,
                    "block_key": str(block_key),
                    "run_file": rel(run_path),
                    "preset": preset_value,
                    "preset_label": run.get("preset_label"),
                    "environment": environment_label,
                    "environment_key": environment_key,
                    "session_stage": session_meta.get("stage"),
                    "started_at": _parse_iso_ts(run.get("started_at")),
                    "ended_at": _parse_iso_ts(run.get("ended_at")),
                    "reason": reason or None,
                    "input_mode": run.get("input_mode"),
                    "step": _to_int(run.get("step")),
                    "speed_dps": _to_float(run.get("speed_dps")),
                    "tolerance": _to_int(run.get("tolerance")),
                    "targets_total": targets_total,
                    "targets_completed": targets_completed,
                    "completion_rate": _safe_div(targets_completed, targets_total) if targets_total else None,
                    "elapsed_total_s": elapsed_total_s,
                    "gestures_total": gestures_total,
                    "gesture_rate_per_s": _safe_div(gestures_total, elapsed_total_s) if gestures_total is not None else None,
                    "seconds_completed_sum": float(sum_completed_seconds),
                    "gestures_completed_sum": int(sum_completed_gestures),
                    "seconds_per_completed_target_mean": seconds_per_completed_target_mean,
                    "gestures_per_completed_target_mean": gestures_per_completed_target_mean,
                    "seconds_current_target_s": time_current_s,
                    "gestures_current_target": gestures_current,
                    "time_first3_s": float(time_first3_s) if targets_completed else None,
                    "gestures_first3": int(gestures_first3) if targets_completed else None,
                    "gesture_events_count": int(gesture_events_count),
                    "target_success_events_count": int(target_success_events_count),
                    "block_completed_at_form": _parse_iso_ts(block_info.get("completed_at")),
                    "block_skipped_form": bool(block_info.get("skipped")) if block_info.get("skipped") is not None else None,
                    "comfort": questionnaire.comfort if questionnaire else None,
                    "performance": questionnaire.performance if questionnaire else None,
                    "control_in_control": questionnaire.in_control if questionnaire else None,
                    "control_expected": questionnaire.expected if questionnaire else None,
                    "control_confident": questionnaire.confident if questionnaire else None,
                    "control_recover": questionnaire.recover if questionnaire else None,
                    "social_self_conscious": questionnaire.self_conscious if questionnaire else None,
                    "social_judged": questionnaire.judged if questionnaire else None,
                    "social_held_back": questionnaire.held_back if questionnaire else None,
                    "justify_social": questionnaire.justified_social if questionnaire else None,
                    "justify_bystanders": questionnaire.justified_bystanders if questionnaire else None,
                    "justify_predict": questionnaire.justified_predict if questionnaire else None,
                    "workload_mental": questionnaire.mental if questionnaire else None,
                    "workload_physical": questionnaire.physical if questionnaire else None,
                    "workload_effort": questionnaire.effort if questionnaire else None,
                    "workload_frustration": questionnaire.frustration if questionnaire else None,
                    "control_confidence_mean": questionnaire.control_confidence_mean if questionnaire else None,
                    "social_mean": questionnaire.social_mean if questionnaire else None,
                    "justification_mean": questionnaire.justification_mean if questionnaire else None,
                    "workload_mean": questionnaire.workload_mean if questionnaire else None,
                    "diagnostic_text": questionnaire.diagnostic if questionnaire else None,
                    "block_questionnaire_completed_at": _parse_iso_ts(run.get("block_questionnaire_completed_at")),
                    "block_questionnaire_skipped": bool(run.get("block_questionnaire_skipped"))
                    if run.get("block_questionnaire_skipped") is not None
                    else None,
                }
            )

            target_map: Dict[int, Dict[str, Optional[int]]] = {}
            for i, t in enumerate(targets):
                if isinstance(t, dict):
                    target_map[i] = {"pan": _to_int(t.get("pan")), "tilt": _to_int(t.get("tilt"))}

            for idx in sorted(completed_indices):
                r = results_by_index.get(int(idx))
                if not isinstance(r, dict):
                    continue
                tgt = target_map.get(int(idx), {})
                targets_rows.append(
                    {
                        "session_id": session_dir.name,
                        "participant_id": participant_id,
                        "block_key": str(block_key),
                        "preset": preset_value,
                        "environment": environment_label,
                        "environment_key": environment_key,
                        "target_index": int(idx),
                        "target_pan": _to_int(r.get("pan")) if r.get("pan") is not None else tgt.get("pan"),
                        "target_tilt": _to_int(r.get("tilt")) if r.get("tilt") is not None else tgt.get("tilt"),
                        "seconds": _to_float(r.get("seconds")),
                        "gestures": _to_int(r.get("gestures")),
                        "completed": True,
                    }
                )

            if reason != "completed" and current_idx is not None and current_idx < targets_total:
                tgt = target_map.get(int(current_idx), {})
                targets_rows.append(
                    {
                        "session_id": session_dir.name,
                        "participant_id": participant_id,
                        "block_key": str(block_key),
                        "preset": preset_value,
                        "environment": environment_label,
                        "environment_key": environment_key,
                        "target_index": int(current_idx),
                        "target_pan": tgt.get("pan"),
                        "target_tilt": tgt.get("tilt"),
                        "seconds": _to_float(run.get("current_target_elapsed")),
                        "gestures": gestures_current,
                        "completed": False,
                    }
                )

        final = questions.get("final") or {}
        if isinstance(final, dict):
            for env_key, env_block in final.items():
                if not isinstance(env_block, dict):
                    continue
                answers = env_block.get("answers") if isinstance(env_block.get("answers"), dict) else {}
                comparison = answers.get("comparison") if isinstance(answers.get("comparison"), dict) else {}
                final_rows.append(
                    {
                        "session_id": session_dir.name,
                        "participant_id": participant_id,
                        "final_env_key": str(env_key),
                        "final_completed_at": _parse_iso_ts(env_block.get("completed_at")),
                        "final_skipped": bool(env_block.get("skipped")) if env_block.get("skipped") is not None else None,
                        "final_answers_skipped": bool(answers.get("skipped")) if answers.get("skipped") is not None else None,
                        "discreet_people": answers.get("discreet_people"),
                        "discreet_locations": answers.get("discreet_locations"),
                        "expressive_people": answers.get("expressive_people"),
                        "expressive_locations": answers.get("expressive_locations"),
                        "prefer_public": _to_int(comparison.get("prefer_public")),
                        "prefer_private": _to_int(comparison.get("prefer_private")),
                        "more_precise": _to_int(comparison.get("more_precise")),
                        "more_embarrassing": _to_int(comparison.get("more_embarrassing")),
                        "more_justified": _to_int(comparison.get("more_justified")),
                    }
                )

    return CollectedRows(
        session_dirs=session_dirs,
        participants=participants_rows,
        blocks=blocks_rows,
        targets=targets_rows,
        gestures=gesture_rows,
        final=final_rows,
        target_direction=target_direction_rows,
        warnings=warnings,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate test-results/* into paper-ready JSON + figures.")
    parser.add_argument("--results-dir", default="test-results", help="Directory containing per-session result folders.")
    parser.add_argument("--out-dir", default="figures", help="Output directory for plots + aggregated JSON.")
    args = parser.parse_args()

    plt, np, pd, sns = _require_analysis_deps()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Results dir not found: {results_dir}", file=sys.stderr)
        return 2

    home_pan, home_tilt = _read_home_pose(Path("server.py"))
    collected = _collect_rows(results_dir, home_pan, home_tilt)

    participants_df = pd.DataFrame(collected.participants)
    blocks_df = pd.DataFrame(collected.blocks)
    targets_df = pd.DataFrame(collected.targets)
    gestures_df = pd.DataFrame(collected.gestures)
    final_df = pd.DataFrame(collected.final)
    target_direction_df = pd.DataFrame(collected.target_direction)

    data_dir = out_dir / "data"

    def df_to_records(df) -> List[Dict[str, Any]]:
        if df.empty:
            return []
        return json.loads(df.to_json(orient="records"))

    _dump_json(data_dir / "participants.json", df_to_records(participants_df))
    _dump_json(data_dir / "blocks.json", df_to_records(blocks_df))
    _dump_json(data_dir / "targets.json", df_to_records(targets_df))
    _dump_json(data_dir / "gestures.json", df_to_records(gestures_df))
    _dump_json(data_dir / "final.json", df_to_records(final_df))
    _dump_json(data_dir / "target_direction.json", df_to_records(target_direction_df))

    summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results_dir": str(results_dir),
        "n_sessions": int(len(collected.session_dirs)),
        "n_participants": int(participants_df["participant_id"].nunique()) if not participants_df.empty else 0,
        "notes": {
            "target_progress_metric": "time_to_targets_progress_by_preset / gestures_to_targets_progress_by_preset show cumulative time/gestures to reach target N (N=1-5). Hatched bars indicate blocks that ended early on the attempted target.",
            "direction_metric": "target_direction.json classifies Arrow-key gestures as towards/away/off-axis vs the start-of-target delta (HOME→target0; target[i-1]→target[i]).",
            "home_pose_used": {"pan": home_pan, "tilt": home_tilt},
            "likert_scale_final_comparison": "1=Discreet, 7=Expressive",
        },
        "warnings": collected.warnings,
    }

    if not blocks_df.empty:
        summary["reason_counts"] = json.loads(
            blocks_df.groupby(["preset", "reason"]).size().reset_index(name="count").to_json(orient="records")
        )

        per_preset: Dict[str, Any] = {}
        for preset, g in blocks_df.groupby("preset", dropna=False):
            preset_key = preset if isinstance(preset, str) and preset else "unknown"
            per_preset[preset_key] = {
                "n_blocks": int(len(g)),
                "n_participants": int(g["participant_id"].nunique()) if "participant_id" in g.columns else None,
                "targets_completed": _describe_numeric(g["targets_completed"]) if "targets_completed" in g.columns else None,
                "completion_rate": _describe_numeric(g["completion_rate"]) if "completion_rate" in g.columns else None,
                "elapsed_total_s": _describe_numeric(g["elapsed_total_s"]) if "elapsed_total_s" in g.columns else None,
                "gestures_total": _describe_numeric(g["gestures_total"]) if "gestures_total" in g.columns else None,
                "time_first3_s": _describe_numeric(g["time_first3_s"]) if "time_first3_s" in g.columns else None,
                "gestures_first3": _describe_numeric(g["gestures_first3"]) if "gestures_first3" in g.columns else None,
                "comfort": _describe_numeric(g["comfort"]) if "comfort" in g.columns else None,
                "performance": _describe_numeric(g["performance"]) if "performance" in g.columns else None,
                "control_confidence_mean": _describe_numeric(g["control_confidence_mean"])
                if "control_confidence_mean" in g.columns
                else None,
                "social_mean": _describe_numeric(g["social_mean"]) if "social_mean" in g.columns else None,
                "justification_mean": _describe_numeric(g["justification_mean"]) if "justification_mean" in g.columns else None,
                "workload_mean": _describe_numeric(g["workload_mean"]) if "workload_mean" in g.columns else None,
            }
        summary["by_preset"] = per_preset

        paired: Dict[str, Any] = {}
        for metric in [
                    "time_first3_s",
                    "gestures_first3",
                    "elapsed_total_s",
                    "gestures_total",
                    "targets_completed",
                    "completion_rate",
                    "comfort",
                    "performance",
                    "control_confidence_mean",
                    "social_mean",
                    "justification_mean",
                    "workload_mean",
                ]:

            if metric not in blocks_df.columns:
                continue
            pivot = (
                blocks_df.pivot_table(
                    index=["participant_id", "environment"],
                    columns="preset",
                    values=metric,
                    aggfunc="mean",
                )
                .reset_index()
                .rename_axis(None, axis=1)
            )
            if "discreet" not in pivot.columns or "expressive" not in pivot.columns:
                continue
            for env, env_df in pivot.groupby("environment"):
                env_clean = env_df.dropna(subset=["discreet", "expressive"])
                if env_clean.empty:
                    continue
                paired_key = f"{metric}__{str(env)}"
                paired[paired_key] = _paired_stats(np, env_clean["discreet"].tolist(), env_clean["expressive"].tolist())
        summary["paired_within_subject"] = paired

    # Interaction-style contrast (difference-of-differences):
    #   (E - D)_public - (E - D)_private
    interaction: Dict[str, Any] = {}
    for metric in [
        "time_first3_s",
        "gestures_first3",
        "elapsed_total_s",
        "gestures_total",
        "targets_completed",
        "completion_rate",
        "comfort",
        "performance",
        "control_confidence_mean",
        "social_mean",
        "justification_mean",
        "workload_mean",
    ]:
        if metric not in blocks_df.columns or "environment_key" not in blocks_df.columns:
            continue
        pivot = (
            blocks_df.pivot_table(
                index=["participant_id", "environment_key"],
                columns="preset",
                values=metric,
                aggfunc="mean",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        if "discreet" not in pivot.columns or "expressive" not in pivot.columns:
            continue
        pivot = pivot.dropna(subset=["discreet", "expressive"])
        if pivot.empty:
            continue
        pivot["diff"] = pivot["expressive"] - pivot["discreet"]  # E-D within each context
        wide = pivot.pivot_table(index="participant_id", columns="environment_key", values="diff", aggfunc="mean").reset_index()
        if "public" not in wide.columns or "private" not in wide.columns:
            continue
        wide_clean = wide.dropna(subset=["public", "private"])
        if wide_clean.empty:
            continue
        stats = _paired_stats(np, wide_clean["private"].tolist(), wide_clean["public"].tolist())
        interaction[metric] = {
            "n": stats.get("n"),
            "public_minus_private_mean": stats.get("mean_diff"),
            "public_minus_private_median": stats.get("median_diff"),
            "public_minus_private_paired_t_p": stats.get("paired_t_p"),
            "public_minus_private_wilcoxon_p": stats.get("wilcoxon_p"),
        }
    if interaction:
        summary["interaction_public_minus_private"] = interaction

    if not targets_df.empty:
        completed_targets = targets_df[targets_df["completed"] == True] 
        if not completed_targets.empty:
            summary["per_target_means"] = json.loads(
                completed_targets.groupby(["preset", "target_index"])[["seconds", "gestures"]]
                .mean(numeric_only=True)
                .reset_index()
                .to_json(orient="records")
            )

    if not gestures_df.empty:
        summary["gesture_inventory"] = json.loads(
            gestures_df.groupby(["preset", "gesture_kind", "gesture_name"])["count"]
            .sum()
            .reset_index()
            .sort_values(["preset", "gesture_kind", "count"], ascending=[True, True, False])
            .to_json(orient="records")
        )

    if not final_df.empty:
        acceptability: Dict[str, Any] = {}
        for key in ["discreet_people", "discreet_locations", "expressive_people", "expressive_locations"]:
            if key not in final_df.columns:
                continue
            counts = Counter()
            for items in final_df[key].dropna().tolist():
                if isinstance(items, list):
                    counts.update([str(x) for x in items])
            acceptability[key] = dict(counts)
        summary["acceptability_counts"] = acceptability

        compare = {}
        for col in ["prefer_public", "prefer_private", "more_precise", "more_embarrassing", "more_justified"]:
            if col in final_df.columns:
                compare[col] = _describe_numeric(final_df[col])
        summary["final_comparison"] = compare

    _dump_json(data_dir / "summary.json", summary)

    plots = _make_plots(plt, np, pd, sns, blocks_df, targets_df, final_df, target_direction_df, out_dir)
    _dump_json(out_dir / "figures_manifest.json", plots)

    print(f"Saved JSON exports to: {data_dir}")
    print(f"Saved plots to: {out_dir / 'plots'}")
    if collected.warnings:
        print(f"Warnings ({len(collected.warnings)}):")
        for w in collected.warnings:
            print(f"  - {w}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

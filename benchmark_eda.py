# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "matplotlib", "seaborn", "numpy", "arabic-reshaper", "python-bidi"]
# ///
"""Exploratory Data Analysis for LLM Benchmark Results."""

import logging
import re
import sys
from pathlib import Path

import arabic_reshaper
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from bidi.algorithm import get_display
from matplotlib.patches import FancyBboxPatch

log = logging.getLogger("benchmark_eda")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "benchmark_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

COMPARISON_CSV = "benchmark_comparison_20260401_091717.csv"
MODEL_CSVS = {
    "Qwen3.5-27B": "benchmark_qwen3.5-27b_20260401_091717.csv",
    "Qwen3.5-9B": "benchmark_qwen3.5-9b_20260401_091717.csv",
    "Qwen3.5-4B": "benchmark_qwen3.5-4b_20260401_091717.csv",
    "Bonsai-8B": "benchmark_bonsai-8b_20260401_091717.csv",
}

# Visual identity
MODEL_COLORS = {
    "Qwen3.5-27B": "#4361ee",
    "Qwen3.5-9B": "#7209b7",
    "Qwen3.5-4B": "#f72585",
    "Bonsai-8B": "#fb8500",
}
MODEL_ORDER = ["Qwen3.5-27B", "Qwen3.5-9B", "Qwen3.5-4B", "Bonsai-8B"]

CATEGORY_LABELS = {
    "general_knowledge": "General\nKnowledge",
    "math": "Math",
    "coding": "Coding",
    "history": "History",
    "logical_reasoning": "Logical\nReasoning",
    "language_understanding": "Language\nUnderstanding",
    "persian": "Persian",
}
CATEGORY_ORDER = list(CATEGORY_LABELS.keys())

DIFFICULTY_ORDER = ["easy", "medium", "hard"]

BG_COLOR = "#0f0f14"
PANEL_COLOR = "#1a1a24"
TEXT_COLOR = "#e0e0e8"
GRID_COLOR = "#2a2a3a"
ACCENT_LIGHT = "#a0a0b8"


def setup_style():
    """Set global matplotlib style for dark, polished look."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": PANEL_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.6,
        "text.color": TEXT_COLOR,
        "xtick.color": ACCENT_LIGHT,
        "ytick.color": ACCENT_LIGHT,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "figure.titlesize": 18,
        "legend.facecolor": PANEL_COLOR,
        "legend.edgecolor": GRID_COLOR,
        "legend.fontsize": 10,
        "legend.labelcolor": TEXT_COLOR,
        "font.family": "sans-serif",
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.facecolor": BG_COLOR,
    })


_RTL_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def fix_rtl(text: str) -> str:
    """Reshape and reorder Arabic/Persian text for correct matplotlib rendering."""
    if not _RTL_RE.search(text):
        return text
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load comparison summary and per-question detail frames."""
    comp = pd.read_csv(DATA_DIR / COMPARISON_CSV)
    log.info("Loaded comparison CSV: %d models", len(comp))

    frames = []
    for model_name, csv_name in MODEL_CSVS.items():
        df = pd.read_csv(DATA_DIR / csv_name)
        df["model"] = model_name
        frames.append(df)
    detail = pd.concat(frames, ignore_index=True)

    # Normalise column name (benchmark runner writes "score_mean", EDA expects "score")
    if "score_mean" in detail.columns and "score" not in detail.columns:
        detail = detail.rename(columns={"score_mean": "score"})

    # Ensure ordered categoricals
    detail["model"] = pd.Categorical(detail["model"], categories=MODEL_ORDER, ordered=True)
    detail["difficulty"] = pd.Categorical(detail["difficulty"], categories=DIFFICULTY_ORDER, ordered=True)
    detail["category"] = pd.Categorical(detail["category"], categories=CATEGORY_ORDER, ordered=True)

    log.info("Loaded detail CSV: %d rows across %d models", len(detail), detail["model"].nunique())
    return comp, detail


# ═══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, name: str):
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    log.info("Saved %s", path)


def _bar_value_labels(ax, fmt="{:.1%}", offset=0.015, fontsize=9):
    """Add value labels on top of bars."""
    for bar in ax.patches:
        h = bar.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center", va="bottom", fontsize=fontsize,
            color=TEXT_COLOR, fontweight="bold",
        )


def _add_subtitle(fig, text, y=0.925, fontsize=11):
    fig.text(0.5, y, text, ha="center", fontsize=fontsize, color=ACCENT_LIGHT, style="italic")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: Overall accuracy comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_overall_accuracy(comp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    models = comp.sort_values("overall_score", ascending=True)
    colors = [MODEL_COLORS[m] for m in models["model"]]

    bars = ax.barh(
        models["model"], models["overall_score"],
        color=colors, edgecolor="white", linewidth=0.5, height=0.55,
        zorder=3,
    )
    for bar, score in zip(bars, models["overall_score"]):
        ax.text(
            score - 0.015, bar.get_y() + bar.get_height() / 2,
            f"{score:.1%}", ha="right", va="center",
            fontsize=13, fontweight="bold", color="white",
        )
    # Add param count annotation
    for bar, (_, row) in zip(bars, models.iterrows()):
        ax.text(
            0.01, bar.get_y() + bar.get_height() / 2,
            f"{row['params_b']}B · {row['quant']}",
            ha="left", va="center", fontsize=9, color="white", alpha=0.7,
        )

    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("")
    ax.set_title("Overall Benchmark Accuracy", fontsize=16, fontweight="bold", pad=14)
    _add_subtitle(fig, "98 questions across 7 categories · 3 difficulty levels")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "01_overall_accuracy")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: Accuracy by category — grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_category_accuracy(detail: pd.DataFrame):
    cat_scores = (
        detail.groupby(["category", "model"], observed=True)["score"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    n_models = len(MODEL_ORDER)
    n_cats = len(CATEGORY_ORDER)
    bar_w = 0.18
    x = np.arange(n_cats)

    for i, model in enumerate(MODEL_ORDER):
        subset = cat_scores[cat_scores["model"] == model].set_index("category").reindex(CATEGORY_ORDER)
        bars = ax.bar(
            x + i * bar_w, subset["score"],
            width=bar_w, label=model, color=MODEL_COLORS[model],
            edgecolor="white", linewidth=0.3, zorder=3,
        )

    ax.set_xticks(x + bar_w * (n_models - 1) / 2)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORY_ORDER], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Category", fontsize=16, fontweight="bold", pad=14)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "02_category_accuracy")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3: Accuracy by difficulty — grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_difficulty_accuracy(detail: pd.DataFrame):
    diff_scores = (
        detail.groupby(["difficulty", "model"], observed=True)["score"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_models = len(MODEL_ORDER)
    bar_w = 0.18
    x = np.arange(len(DIFFICULTY_ORDER))

    for i, model in enumerate(MODEL_ORDER):
        subset = diff_scores[diff_scores["model"] == model].set_index("difficulty").reindex(DIFFICULTY_ORDER)
        ax.bar(
            x + i * bar_w, subset["score"],
            width=bar_w, label=model, color=MODEL_COLORS[model],
            edgecolor="white", linewidth=0.3, zorder=3,
        )

    ax.set_xticks(x + bar_w * (n_models - 1) / 2)
    ax.set_xticklabels([d.capitalize() for d in DIFFICULTY_ORDER], fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Difficulty Level", fontsize=16, fontweight="bold", pad=14)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "03_difficulty_accuracy")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4: Radar / spider chart — category profile per model
# ═══════════════════════════════════════════════════════════════════════════════

def plot_radar(detail: pd.DataFrame):
    cat_scores = (
        detail.groupby(["category", "model"], observed=True)["score"]
        .mean()
        .reset_index()
    )
    categories = CATEGORY_ORDER
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.set_facecolor(PANEL_COLOR)

    for model in MODEL_ORDER:
        vals = []
        for cat in categories:
            row = cat_scores[(cat_scores["model"] == model) & (cat_scores["category"] == cat)]
            vals.append(row["score"].values[0] if len(row) else 0)
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=model, color=MODEL_COLORS[model], markersize=5)
        ax.fill(angles, vals, alpha=0.08, color=MODEL_COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([CATEGORY_LABELS[c].replace("\n", " ") for c in categories], fontsize=10, color=TEXT_COLOR)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color=ACCENT_LIGHT)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.4)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.4)
    ax.spines["polar"].set_color(GRID_COLOR)
    ax.set_title("Category Profile (Radar)", fontsize=16, fontweight="bold", pad=24, color=TEXT_COLOR)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), framealpha=0.9)
    fig.tight_layout()
    _save(fig, "04_radar_category")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5: Generation speed (tok/s) comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_speed_comparison(comp: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Gen tok/s
    ax = axes[0]
    models = comp.set_index("model").reindex(MODEL_ORDER)
    bars = ax.bar(
        MODEL_ORDER, models["avg_gen_tok_s"],
        color=[MODEL_COLORS[m] for m in MODEL_ORDER],
        edgecolor="white", linewidth=0.3, width=0.55, zorder=3,
    )
    for bar, val in zip(bars, models["avg_gen_tok_s"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylabel("Tokens / second")
    ax.set_title("Generation Speed", fontsize=14, fontweight="bold")
    ax.grid(axis="x", visible=False)

    # Prompt tok/s
    ax = axes[1]
    bars = ax.bar(
        MODEL_ORDER, models["avg_prompt_tok_s"],
        color=[MODEL_COLORS[m] for m in MODEL_ORDER],
        edgecolor="white", linewidth=0.3, width=0.55, zorder=3,
    )
    for bar, val in zip(bars, models["avg_prompt_tok_s"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 8,
                f"{val:.0f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylabel("Tokens / second")
    ax.set_title("Prompt Processing Speed", fontsize=14, fontweight="bold")
    ax.grid(axis="x", visible=False)

    fig.suptitle("Inference Speed Comparison", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "05_speed_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 6: Accuracy vs Speed trade-off scatter
# ═══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_vs_speed(comp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in comp.iterrows():
        model = row["model"]
        ax.scatter(
            row["avg_gen_tok_s"], row["overall_score"],
            s=row["params_b"] * 18,  # bubble size proportional to params
            color=MODEL_COLORS[model], edgecolors="white", linewidth=1.2,
            zorder=5, alpha=0.9,
        )
        ax.annotate(
            f"{model}\n({row['params_b']}B)",
            (row["avg_gen_tok_s"], row["overall_score"]),
            textcoords="offset points", xytext=(12, -5),
            fontsize=10, color=MODEL_COLORS[model], fontweight="bold",
        )

    ax.set_xlabel("Generation Speed (tok/s)")
    ax.set_ylabel("Overall Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Accuracy vs. Generation Speed", fontsize=16, fontweight="bold", pad=14)
    _add_subtitle(fig, "Bubble size proportional to parameter count")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "06_accuracy_vs_speed")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 7: Total wall time
# ═══════════════════════════════════════════════════════════════════════════════

def plot_wall_time(comp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    models = comp.sort_values("total_wall_time_s", ascending=True)
    colors = [MODEL_COLORS[m] for m in models["model"]]

    bars = ax.barh(
        models["model"], models["total_wall_time_s"],
        color=colors, edgecolor="white", linewidth=0.5, height=0.55, zorder=3,
    )
    for bar, val in zip(bars, models["total_wall_time_s"]):
        minutes = val / 60
        ax.text(
            val + 3, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}s ({minutes:.1f}m)",
            ha="left", va="center", fontsize=11, fontweight="bold", color=TEXT_COLOR,
        )

    ax.set_xlabel("Total Wall Time (seconds)")
    ax.set_title("Total Benchmark Wall Time", fontsize=16, fontweight="bold", pad=14)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, "07_wall_time")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 8: Per-question heatmap — score agreement across models
# ═══════════════════════════════════════════════════════════════════════════════

def plot_question_heatmap(detail: pd.DataFrame):
    pivot = detail.pivot_table(
        index="question_id", columns="model", values="score",
    ).reindex(columns=MODEL_ORDER)

    # Add category info for row labels
    q_meta = detail[["question_id", "category", "difficulty"]].drop_duplicates().set_index("question_id")
    pivot = pivot.join(q_meta)
    pivot = pivot.sort_values(["category", "difficulty", "question_id"])
    row_labels = [
        f"Q{qid} ({CATEGORY_LABELS[cat].replace(chr(10), ' ')[:12]}, {diff[0].upper()})"
        for qid, cat, diff in zip(pivot.index, pivot["category"], pivot["difficulty"])
    ]
    score_matrix = pivot[MODEL_ORDER].values

    fig, ax = plt.subplots(figsize=(8, 18))
    cmap = sns.color_palette("blend:#1a1a24,#4361ee,#00e676", as_cmap=True)
    im = ax.imshow(score_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title("Per-Question Score Heatmap", fontsize=16, fontweight="bold", pad=14)

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Score", color=TEXT_COLOR)
    cbar.ax.yaxis.set_tick_params(color=ACCENT_LIGHT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=ACCENT_LIGHT)

    fig.tight_layout()
    _save(fig, "08_question_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 9: Generation speed distribution (box + strip)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_speed_distribution(detail: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    palette = {m: MODEL_COLORS[m] for m in MODEL_ORDER}
    sns.boxplot(
        data=detail, x="model", y="gen_tok_s", order=MODEL_ORDER,
        palette=palette, width=0.45, linewidth=1.2,
        fliersize=0, boxprops={"alpha": 0.7}, ax=ax, zorder=3,
    )
    sns.stripplot(
        data=detail, x="model", y="gen_tok_s", order=MODEL_ORDER,
        palette=palette, size=4, alpha=0.4, jitter=0.15, ax=ax, zorder=4,
    )

    ax.set_ylabel("Generation Speed (tok/s)")
    ax.set_xlabel("")
    ax.set_title("Generation Speed Distribution per Model", fontsize=16, fontweight="bold", pad=14)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "09_speed_distribution")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 10: Response verbosity (completion tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_verbosity(detail: pd.DataFrame):
    # completion_tokens may be 0 when llama.cpp streaming doesn't report usage;
    # fall back to character length of the full_response column.
    col = "completion_tokens"
    if detail[col].sum() == 0 and "full_response" in detail.columns:
        detail = detail.copy()
        detail["_resp_len"] = detail["full_response"].fillna("").str.len()
        col = "_resp_len"
        ylabel = "Response Length (chars)"
    else:
        ylabel = "Completion Tokens"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Median per model
    ax = axes[0]
    medians = detail.groupby("model", observed=True)[col].median().reindex(MODEL_ORDER)
    bars = ax.bar(
        MODEL_ORDER, medians,
        color=[MODEL_COLORS[m] for m in MODEL_ORDER],
        edgecolor="white", linewidth=0.3, width=0.55, zorder=3,
    )
    for bar, val in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(medians) * 0.02,
                f"{val:.0f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylabel(ylabel)
    ax.set_title("Response Length (Median)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", visible=False)

    # Box plot
    ax = axes[1]
    palette = {m: MODEL_COLORS[m] for m in MODEL_ORDER}
    sns.boxplot(
        data=detail, x="model", y=col, order=MODEL_ORDER,
        hue="model", palette=palette, legend=False,
        width=0.45, linewidth=1.2,
        flierprops={"markerfacecolor": ACCENT_LIGHT, "markersize": 3},
        boxprops={"alpha": 0.7}, ax=ax, zorder=3,
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_title("Response Length Distribution", fontsize=14, fontweight="bold")
    ax.grid(axis="x", visible=False)

    fig.suptitle("Response Verbosity", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "10_verbosity")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 11: Accuracy vs Model Size
# ═══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_vs_size(comp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in comp.iterrows():
        model = row["model"]
        ax.scatter(
            row["params_b"], row["overall_score"],
            s=250, color=MODEL_COLORS[model], edgecolors="white",
            linewidth=1.5, zorder=5, alpha=0.95,
        )
        ax.annotate(
            f"{model}\n{row['quant']}",
            (row["params_b"], row["overall_score"]),
            textcoords="offset points", xytext=(14, 0),
            fontsize=10, color=MODEL_COLORS[model], fontweight="bold",
        )

    ax.set_xlabel("Parameters (Billions)")
    ax.set_ylabel("Overall Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Accuracy vs. Model Size", fontsize=16, fontweight="bold", pad=14)
    _add_subtitle(fig, "Quantization level noted — Bonsai-8B uses aggressive Q1_0")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "11_accuracy_vs_size")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 12: Efficiency — accuracy per billion parameters
# ═══════════════════════════════════════════════════════════════════════════════

def plot_efficiency(comp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    comp = comp.copy()
    comp["efficiency"] = comp["overall_score"] / comp["params_b"]
    comp = comp.sort_values("efficiency", ascending=True)

    colors = [MODEL_COLORS[m] for m in comp["model"]]
    bars = ax.barh(
        comp["model"], comp["efficiency"],
        color=colors, edgecolor="white", linewidth=0.5, height=0.55, zorder=3,
    )
    for bar, val in zip(bars, comp["efficiency"]):
        ax.text(
            val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", ha="left", va="center",
            fontsize=11, fontweight="bold", color=TEXT_COLOR,
        )

    ax.set_xlabel("Accuracy / Billion Parameters")
    ax.set_title("Accuracy Efficiency (Score per Billion Params)", fontsize=16, fontweight="bold", pad=14)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, "12_efficiency")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 13: Difficulty breakdown heatmap (model × category × difficulty)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_difficulty_category_heatmap(detail: pd.DataFrame):
    agg = (
        detail.groupby(["model", "category", "difficulty"], observed=True)["score"]
        .mean()
        .reset_index()
    )
    # Create composite label: category + difficulty
    agg["cat_diff"] = agg["category"].astype(str) + " · " + agg["difficulty"].astype(str)

    # Pivot for heatmap
    pivot = agg.pivot_table(index="cat_diff", columns="model", values="score")
    # Sort by category order, then difficulty
    order = []
    for cat in CATEGORY_ORDER:
        for diff in DIFFICULTY_ORDER:
            label = f"{cat} · {diff}"
            if label in pivot.index:
                order.append(label)
    pivot = pivot.reindex(index=order, columns=MODEL_ORDER)

    # Prettify row labels
    pretty_labels = []
    for label in pivot.index:
        cat, diff = label.split(" · ")
        pretty_labels.append(f"{CATEGORY_LABELS[cat].replace(chr(10), ' ')} ({diff[0].upper()})")

    fig, ax = plt.subplots(figsize=(9, 12))
    cmap = sns.color_palette("blend:#2d1b3d,#7209b7,#00e676", as_cmap=True)
    sns.heatmap(
        pivot, annot=True, fmt=".0%", cmap=cmap,
        linewidths=1.5, linecolor=BG_COLOR,
        cbar_kws={"label": "Accuracy", "shrink": 0.5},
        ax=ax, vmin=0, vmax=1,
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )
    ax.set_yticklabels(pretty_labels, fontsize=9, rotation=0)
    ax.set_xticklabels(MODEL_ORDER, fontsize=10, rotation=30, ha="right")
    ax.set_title("Accuracy: Category × Difficulty × Model", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    _save(fig, "13_difficulty_category_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 14: Pairwise model agreement
# ═══════════════════════════════════════════════════════════════════════════════

def plot_model_agreement(detail: pd.DataFrame):
    """How often do two models give the same correct/incorrect result?"""
    pivot = detail.pivot_table(index="question_id", columns="model", values="score")
    pivot = pivot.reindex(columns=MODEL_ORDER)

    # Binarize: fully correct (1.0) vs not
    binary = (pivot >= 0.99).astype(int)

    n = len(MODEL_ORDER)
    agreement = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            agreement[i, j] = (binary.iloc[:, i] == binary.iloc[:, j]).mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = sns.color_palette("blend:#1a1a24,#4361ee,#00e676", as_cmap=True)
    sns.heatmap(
        pd.DataFrame(agreement, index=MODEL_ORDER, columns=MODEL_ORDER),
        annot=True, fmt=".0%", cmap=cmap, vmin=0.5, vmax=1.0,
        linewidths=2, linecolor=BG_COLOR,
        cbar_kws={"label": "Agreement Rate", "shrink": 0.7},
        ax=ax, annot_kws={"fontsize": 12, "fontweight": "bold"},
    )
    ax.set_title("Pairwise Model Agreement", fontsize=16, fontweight="bold", pad=14)
    _add_subtitle(fig, "How often two models agree on correct/incorrect (binary)")
    ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(MODEL_ORDER, rotation=0, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "14_model_agreement")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 15: Hardest questions — which questions stump most models?
# ═══════════════════════════════════════════════════════════════════════════════

def plot_hardest_questions(detail: pd.DataFrame):
    avg_score = detail.groupby("question_id")["score"].mean().reset_index()
    avg_score = avg_score.sort_values("score").head(15)

    q_meta = detail[["question_id", "category", "difficulty", "prompt_preview"]].drop_duplicates()
    avg_score = avg_score.merge(q_meta, on="question_id")
    avg_score["label"] = avg_score.apply(
        lambda r: fix_rtl(f"Q{r['question_id']}: {r['prompt_preview'][:50]}…"), axis=1
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [
        "#e63946" if s < 0.5 else "#fca311" if s < 0.75 else "#00b4d8"
        for s in avg_score["score"]
    ]
    bars = ax.barh(
        avg_score["label"], avg_score["score"],
        color=colors, edgecolor="white", linewidth=0.3, height=0.7, zorder=3,
    )
    for bar, val in zip(bars, avg_score["score"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", ha="left", va="center",
                fontsize=10, fontweight="bold", color=TEXT_COLOR)

    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Average Score Across Models")
    ax.set_title("Hardest Questions (Lowest Avg Score)", fontsize=16, fontweight="bold", pad=14)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, "15_hardest_questions")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 16: Speed vs Difficulty
# ═══════════════════════════════════════════════════════════════════════════════

def plot_speed_by_difficulty(detail: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    palette = {m: MODEL_COLORS[m] for m in MODEL_ORDER}
    agg = (
        detail.groupby(["model", "difficulty"], observed=True)["wall_time_ms"]
        .median()
        .reset_index()
    )
    agg["wall_time_s"] = agg["wall_time_ms"] / 1000

    n_models = len(MODEL_ORDER)
    bar_w = 0.18
    x = np.arange(len(DIFFICULTY_ORDER))

    for i, model in enumerate(MODEL_ORDER):
        subset = agg[agg["model"] == model].set_index("difficulty").reindex(DIFFICULTY_ORDER)
        ax.bar(
            x + i * bar_w, subset["wall_time_s"],
            width=bar_w, label=model, color=MODEL_COLORS[model],
            edgecolor="white", linewidth=0.3, zorder=3,
        )

    ax.set_xticks(x + bar_w * (n_models - 1) / 2)
    ax.set_xticklabels([d.capitalize() for d in DIFFICULTY_ORDER], fontsize=12)
    ax.set_ylabel("Median Wall Time (seconds)")
    ax.set_title("Response Time by Difficulty", fontsize=16, fontweight="bold", pad=14)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "16_speed_by_difficulty")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 17: Summary dashboard — key metrics table
# ═══════════════════════════════════════════════════════════════════════════════

def plot_summary_table(comp: pd.DataFrame, detail: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    comp_sorted = comp.set_index("model").reindex(MODEL_ORDER)

    col_headers = ["Model", "Params", "Quant", "Accuracy", "Gen tok/s", "Prompt tok/s", "Wall Time", "Efficiency"]
    rows = []
    for model in MODEL_ORDER:
        r = comp_sorted.loc[model]
        eff = r["overall_score"] / r["params_b"]
        rows.append([
            model,
            f"{r['params_b']:.1f}B",
            r["quant"],
            f"{r['overall_score']:.1%}",
            f"{r['avg_gen_tok_s']:.1f}",
            f"{r['avg_prompt_tok_s']:.0f}",
            f"{r['total_wall_time_s']:.0f}s",
            f"{eff:.4f}",
        ])

    table = ax.table(
        cellText=rows, colLabels=col_headers,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(col_headers)):
        cell = table[0, j]
        cell.set_facecolor("#2a2a3a")
        cell.set_text_props(color=TEXT_COLOR, fontweight="bold", fontsize=11)
        cell.set_edgecolor(GRID_COLOR)

    # Style data cells
    for i in range(len(MODEL_ORDER)):
        for j in range(len(col_headers)):
            cell = table[i + 1, j]
            cell.set_facecolor(PANEL_COLOR)
            cell.set_text_props(color=TEXT_COLOR, fontsize=10)
            cell.set_edgecolor(GRID_COLOR)
            if j == 0:
                cell.set_text_props(color=MODEL_COLORS[MODEL_ORDER[i]], fontweight="bold", fontsize=11)

    ax.set_title("Benchmark Summary", fontsize=18, fontweight="bold", pad=20, color=TEXT_COLOR)
    fig.tight_layout()
    _save(fig, "00_summary_table")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    setup_style()
    comp, detail = load_data()

    log.info("Generating plots…")
    plot_summary_table(comp, detail)
    plot_overall_accuracy(comp)
    plot_category_accuracy(detail)
    plot_difficulty_accuracy(detail)
    plot_radar(detail)
    plot_speed_comparison(comp)
    plot_accuracy_vs_speed(comp)
    plot_wall_time(comp)
    plot_question_heatmap(detail)
    plot_speed_distribution(detail)
    plot_verbosity(detail)
    plot_accuracy_vs_size(comp)
    plot_efficiency(comp)
    plot_difficulty_category_heatmap(detail)
    plot_model_agreement(detail)
    plot_hardest_questions(detail)
    plot_speed_by_difficulty(detail)

    log.info("All %d plots saved to %s", 17, OUTPUT_DIR)


if __name__ == "__main__":
    main()

"""Figures for the write-up, drawn from the tracked results CSVs.

Run from the repo root:  uv run --group writeup python write-up/make_figures.py

Writes vector PDFs to write-up/figures/:
  price-paths.pdf       selling price by period, one panel per model (small-model survey, n = 4, NT-T)
  talk-vs-notalk.pdf    NoTalk vs Talk selling price per model, periods 6+, against humans at n = 4
  q35-replication.pdf   Qwen3.5-9B per-market NoTalk -> Talk at n = 2 and n = 4, against humans

Selling price = market price (lowest posted). Part means use periods 6+ of each part (sp_p1_from6, sp_p2_from6).
Human benchmarks come from the human_comparison.csv files (Fonseca & Normann 2012).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import pandas as pd

from oligopoly import benchmarks as b

ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / "results/2026-10-21-small-models/analysis"
REPL = ROOT / "results/2026-10-22-q35-9b-replication/analysis"
OUT = ROOT / "write-up/figures"

# Categorical slots 1-2 of the dataviz reference palette (validated light, adjacent CVD dE 24.7).
NOTALK, TALK = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MEAN = "#6250d6"  # slot 7 (violet): the Qwen3.5-9B mean in the replication figure, apart from the Part colors
WIDTH = 6.5  # inches, LaTeX text width

# Display name and parameter count (total / active) for each model in the survey.
MODELS = {
    "qwen3.8-27b": ("Qwen3.8-27B", "27B dense"),
    "qwen3.6-27b": ("Qwen3.6-27B", "27B dense"),
    "qwen3.5-27b": ("Qwen3.5-27B", "27B dense"),
    "bonsai-2-27b": ("Bonsai-2-27B", "27B dense, ternary"),
    "gemma-4-26b-a4b": ("Gemma-4-26B-A4B", "26B / 4B MoE"),
    "gpt-oss-20b": ("gpt-oss-20b", "21B / 3.6B MoE"),
    "qwen3-14b": ("Qwen3-14B", "14B dense"),
    "qwen3.5-9b": ("Qwen3.5-9B", "9B dense"),
    "granite-4.2-8b": ("Granite-4.2-8B", "8B dense"),
    "lfm-2.5-2.6b": ("LFM-2.5-2.6B", "2.6B dense"),
}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "pdf.fonttype": 42,
    "figure.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def humans(n):
    """Human (NoTalk, Talk) mean selling price at n firms (Fonseca & Normann, NT-T cell)."""
    return b.MAIN_NOTALK_SELLING[n].mean, b.MAIN_TALK_SELLING[n].mean


def survey_order():
    """Survey markets sorted by Talk price, then NoTalk price, highest first."""
    m = pd.read_csv(SURVEY / "markets.csv")
    return m.sort_values(["sp_p2_from6", "sp_p1_from6"], ascending=False)


def price_paths():
    mp = pd.read_csv(SURVEY / "market_period.csv")
    h_nt, h_t = humans(4)
    order = survey_order()
    fig, axes = plt.subplots(2, 5, figsize=(WIDTH, 3.3), sharex=True, sharey=True)
    for ax, row in zip(axes.flat, order.itertuples()):
        d = mp[mp.market_id == row.market_id].sort_values("gperiod")
        p1 = row.part1_len
        end = d.gperiod.max()
        ax.axvline(p1 + 0.5, color=MUTED, lw=0.6)
        ax.hlines([h_nt, h_t], [1, p1 + 1], [p1, end], colors=INK, lw=0.8, linestyles=(0, (3, 2)))
        for part, color in ((1, NOTALK), (2, TALK)):
            s = d[d.part == part]
            ax.plot(s.gperiod, s.sp, color=color, lw=1.4, solid_capstyle="round")
        name, params = MODELS[row.model]
        ax.set_title(f"{name}\n{params}", color=INK, pad=3)
        ax.set_ylim(-4, 104)
        ax.set_yticks([0, 50, 100])
        ax.set_xticks([1, p1 + 1, end])
        ax.set_xlim(0, end + 1)
    for ax in axes[1]:
        ax.set_xlabel("Period")
    for ax in axes[:, 0]:
        ax.set_ylabel("Selling price")
    handles = [
        plt.Line2D([], [], color=NOTALK, lw=1.4, label="NoTalk (Part 1)"),
        plt.Line2D([], [], color=TALK, lw=1.4, label="Talk (Part 2)"),
        plt.Line2D([], [], color=INK, lw=0.8, ls=(0, (3, 2)), label=f"Humans, n = 4 (NoTalk {h_nt:g}, Talk {h_t:g})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.8, w_pad=0.4)
    fig.savefig(OUT / "price-paths.pdf")
    plt.close(fig)


def talk_vs_notalk():
    order = survey_order()
    rep = pd.read_csv(REPL / "markets.csv")
    rep4 = rep[(rep.n == 4) & (rep.model == "qwen3.5-9b")]
    h_nt, h_t = humans(4)

    rows = [(f"Humans ({b.MARKETS_PER_CELL} markets)", h_nt, h_t, None)]
    for r in order.itertuples():
        name = MODELS[r.model][0]
        if r.model == "qwen3.5-9b":
            rows.append((f"{name} (mean of {len(rep4)} markets)", rep4.sp_p1_from6.mean(), rep4.sp_p2_from6.mean(), rep4))
        else:
            rows.append((name, r.sp_p1_from6, r.sp_p2_from6, None))
    # Re-sort the model rows by Talk price now that Qwen3.5-9B shows its replicated mean.
    rows = rows[:1] + sorted(rows[1:], key=lambda r: (-r[2], -r[1]))

    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    for y, (label, nt, t, markets) in enumerate(rows):
        ax.plot([nt, t], [y, y], color=GRID if markets is None else "#c3c2b7", lw=2.2, zorder=1,
                solid_capstyle="round")
        if markets is not None:
            ax.scatter(markets.sp_p1_from6, [y - 0.22] * len(markets), s=9, color=NOTALK, alpha=0.6, lw=0)
            ax.scatter(markets.sp_p2_from6, [y + 0.22] * len(markets), s=9, color=TALK, alpha=0.6, lw=0)
        human = y == 0
        kw = dict(s=34, zorder=3, edgecolors="white", linewidths=0.8)
        dy = 0.14 if abs(nt - t) < 2 else 0  # split coincident points (e.g. both at 100) so both stay visible
        ax.scatter(nt, y - dy, color=INK if human else NOTALK, marker="o", **kw)
        ax.scatter(t, y + dy, color=INK if human else TALK, marker="D" if human else "o", **kw)
    ax.axvline(h_nt, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.axvline(h_t, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(-3, 103)
    ax.set_xlabel("Mean selling price, periods 6+ of each part (n = 4)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    handles = [
        plt.Line2D([], [], color=NOTALK, marker="o", ls="", ms=5, label="NoTalk"),
        plt.Line2D([], [], color=TALK, marker="o", ls="", ms=5, label="Talk"),
        plt.Line2D([], [], color=INK, marker="o", ls="", ms=5, label="Humans NoTalk"),
        plt.Line2D([], [], color=INK, marker="D", ls="", ms=4.5, label="Humans Talk"),
        (plt.Line2D([], [], color=NOTALK, marker="o", ls="", ms=3, alpha=0.6),
         plt.Line2D([], [], color=TALK, marker="o", ls="", ms=3, alpha=0.6)),
    ]
    labels = [h.get_label() for h in handles[:4]] + ["Single Qwen3.5-9B markets"]
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=5, frameon=False,
              handletextpad=0.3, columnspacing=1.2)
    ax.text(1.0, -0.17, "Other models: one market each", transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "talk-vs-notalk.pdf")
    plt.close(fig)


def q35_replication():
    rep = pd.read_csv(REPL / "markets.csv")
    rep = rep[rep.model == "qwen3.5-9b"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True)
    for ax, n in zip(axes, (2, 4)):
        d = rep[rep.n == n]
        h_nt, h_t = humans(n)
        for r in d.itertuples():
            survey = r.run_id.startswith("q35-9b-n4-nt-t-2026-10-21")
            ax.plot([0, 1], [r.sp_p1_from6, r.sp_p2_from6], color=MUTED if survey else "#b4b3ad",
                    lw=1.0, marker="o", ms=3.5, zorder=2)
            if survey:
                ax.annotate("survey", (0, r.sp_p1_from6), xytext=(-6, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=7, color=MUTED)
        ax.plot([0, 1], [d.sp_p1_from6.mean(), d.sp_p2_from6.mean()], color=MEAN, lw=2.2, marker="o",
                ms=6, zorder=3, mec="white", mew=0.8)
        ax.plot([0, 1], [h_nt, h_t], color=INK, lw=1.4, ls=(0, (3, 2)), marker="D", ms=5, zorder=3,
                mec="white", mew=0.8)
        ax.annotate(f"{h_nt:.1f}", (0, h_nt), xytext=(-6, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=7, color=INK)
        ax.annotate(f"{h_t:.1f}", (1, h_t), xytext=(6, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=7, color=INK)
        ax.set_title(f"n = {n}  ({len(d)} markets)", color=INK)
        ax.set_xticks([0, 1], ["NoTalk", "Talk"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(-3, 104)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean selling price, periods 6+")
    handles = [
        plt.Line2D([], [], color="#b4b3ad", marker="o", ms=3.5, lw=1.0, label="One Qwen3.5-9B market"),
        plt.Line2D([], [], color=MEAN, marker="o", ms=5, lw=2.2, label="Qwen3.5-9B mean"),
        plt.Line2D([], [], color=INK, marker="D", ms=4.5, lw=1.4, ls=(0, (3, 2)), label=f"Humans (mean of {b.MARKETS_PER_CELL})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "q35-replication.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    price_paths()
    talk_vs_notalk()
    q35_replication()
    print(f"wrote {', '.join(p.name for p in sorted(OUT.glob('*.pdf')))} to {OUT}")

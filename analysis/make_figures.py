"""Draw the four figures of the paper into analysis/figures (300 dpi). Fig. 3 and Fig. 4 read analysis/data.json,
written by extract_results.py.
  python analysis/make_figures.py
"""
# ---- Fig. 1 and Fig. 2 (from make_figures_v0.7.py)
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 9})

AGENT, SYS, GATE, SIDE, EDGE = "#DCE8F7", "#EFEFEF", "#FBE3C8", "#E6F2E1", "#333333"
GATE_TEXT = ("Bounded-autonomy gate\nT0/T1 free  |  T2/T3 only if in the approved plan  |  "
             "no data destruction  |  privileged access only if birthright")


def box(ax, x, y, w, h, text, fill, bold=False, fs=9, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=EDGE, lw=1, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, style="-|>", ls="-", color=EDGE, both=False, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="<|-|>" if both else style,
                                 mutation_scale=10, lw=1, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


XS = [0.9, 3.6, 6.3, 9.0]


def systems_row(ax, y_gate, y_sys):
    for x, name in zip(XS, ["Identity", "Workday HCM", "Finance", "ITSM / Legal"]):
        arrow(ax, x + 1.05, y_gate, x + 1.05, y_sys + 0.8)
        box(ax, x, y_sys, 2.1, 0.8, name, SYS, bold=True)


def panel_single(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.text(0.05, 4.45, "(a) Single agent", fontsize=10, fontweight="bold", va="top")
    box(ax, 0.2, 2.95, 2.4, 0.9, "Workday HCM\nevent feed", SYS, bold=True)
    box(ax, 3.4, 2.8, 5.2, 1.2, "Single agent\ninvestigates, plans, executes and verifies\n(all tools, one conversation)",
        AGENT, fs=8.5)
    arrow(ax, 2.6, 3.4, 3.4, 3.4)
    box(ax, 9.2, 2.8, 2.6, 1.2, "Owner teams and\nhuman escalation", SIDE, fs=8)
    arrow(ax, 8.6, 3.4, 9.2, 3.4, both=True)
    arrow(ax, 6.0, 2.8, 6.0, 2.45)
    ax.add_patch(FancyBboxPatch((0.6, 1.55), 10.8, 0.85, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec="#999999", lw=1, ls="--"))
    ax.text(6.0, 1.97, "Bounded-autonomy gate (not used in the evaluated single-agent configuration)",
            ha="center", va="center", fontsize=7.6, color="#666666", style="italic")
    systems_row(ax, 1.55, 0.35)


def panel_multi(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.9)
    ax.axis("off")
    ax.text(0.05, 7.85, "(b) Multi-agent team", fontsize=10, fontweight="bold", va="top")
    box(ax, 0.2, 6.3, 2.4, 0.9, "Workday HCM\nevent feed", SYS, bold=True)
    box(ax, 3.1, 6.3, 2.2, 0.9, "Investigator\n(read-only)", AGENT)
    box(ax, 5.9, 6.3, 1.9, 0.9, "Planner", AGENT, bold=True)
    box(ax, 8.5, 6.3, 1.9, 0.9, "Critic", AGENT)
    arrow(ax, 2.6, 6.75, 3.1, 6.75)
    arrow(ax, 5.3, 6.75, 5.9, 6.75)
    arrow(ax, 7.8, 6.75, 8.5, 6.75, both=True)
    ax.text(8.15, 7.3, "review / revise", ha="center", fontsize=7.5, style="italic")
    arrow(ax, 6.85, 6.3, 6.85, 5.25)
    ax.text(7.0, 5.72, "approved plan, run in its own order", ha="left", fontsize=7.5, style="italic")
    for x, name in zip(XS, ["Identity\nExecutor", "Workday HCM\nExecutor", "Finance\nExecutor", "ITSM\nExecutor"]):
        box(ax, x, 4.35, 2.1, 0.85, name, AGENT)
        arrow(ax, x + 1.05, 4.35, x + 1.05, 4.05)
    ax.add_patch(FancyBboxPatch((0.6, 3.15), 10.8, 0.9, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=GATE, ec=EDGE, lw=1, ls="--"))
    ax.text(6.0, 3.6, GATE_TEXT, ha="center", va="center", fontsize=7.2)
    systems_row(ax, 3.15, 1.95)
    ax.plot([1.95, 10.05], [1.7, 1.7], color=EDGE, lw=1)
    for x in XS:
        ax.plot([x + 1.05, x + 1.05], [1.95, 1.7], color=EDGE, lw=1)
    arrow(ax, 3.2, 1.7, 3.2, 1.3)
    box(ax, 0.6, 0.15, 5.2, 1.15, "Reconciler (read-only)\nchecks that all systems agree;\n"
                                  "discrepancies go back to the Planner", AGENT, fs=7.8)
    box(ax, 6.4, 0.15, 5.0, 1.15, "Owner teams (Finance, Identity, HR)\nand human escalation:\n"
                                  "precise fix request, then reapply", SIDE, fs=7.8)


def figure1():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 7.6), gridspec_kw={"height_ratios": [4.6, 7.9]})
    panel_single(a1)
    panel_multi(a2)
    fig.tight_layout(h_pad=0.6)
    path = os.path.join(HERE, "figures", "fig1_configurations.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def figure2():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9.4)
    ax.axis("off")
    lanes = {"Workday HCM": 1.3, "Agentic\norchestrator": 4.4, "Identity": 7.5, "ITSM / Finance": 10.5}
    for name, x in lanes.items():
        box(ax, x - 1.0, 8.4, 2.0, 0.8, name, SYS if "orch" not in name else AGENT, bold=True, fs=8.5)
        ax.plot([x, x], [0.3, 8.4], color="#888888", lw=0.8, ls="--")

    def msg(y, a, b, text, ls="-", color=EDGE):
        xa, xb = lanes[a], lanes[b]
        arrow(ax, xa, y, xb, y, ls=ls, color=color)
        ax.text((xa + xb) / 2, y + 0.12, text, ha="center", va="bottom", fontsize=7.5, color=color)

    W, O, I, T = "Workday HCM", "Agentic\norchestrator", "Identity", "ITSM / Finance"
    msg(7.7, W, O, "1. HIRE event (case ID)")
    msg(6.9, O, I, "2. create account (idempotency key)")
    msg(6.3, I, O, "UPN / email", ls="--")
    msg(5.5, O, W, "3. write back email (case ID)")
    msg(4.9, W, O, "4. business process: completed / ERROR", ls="--")
    ax.add_patch(FancyBboxPatch((2.4, 3.85), 4.2, 0.65, boxstyle="round,pad=0.02", fc="#FDECEA", ec="#B03A2E", lw=1))
    ax.text(4.5, 4.17, "if ERROR (writeback lost): reapply\nonly the writeback; never recreate the account",
            ha="center", va="center", fontsize=7.3, color="#7B241C")
    msg(3.2, W, O, "5. WORKER_CONTACT_CHANGED (same case ID)", ls="--")
    msg(2.4, O, T, "6. finance profile + equipment order")
    msg(1.6, T, O, "confirmed", ls="--")
    msg(0.8, O, W, "7. onboarding tasks complete")
    fig.tight_layout()
    path = os.path.join(HERE, "figures", "fig2_closed_loop.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path



_fig1, _fig2 = figure1, figure2

# ---- Fig. 3 and Fig. 4 (from make_figures_v0.6.py)
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "Arial", "font.size": 9})
D = json.load(open(os.path.join(HERE, "data.json"), encoding="utf-8"))

ORDER = ["single_agent", "multi_agent_unguarded", "multi_agent"]
LABEL = {"single_agent": "Single\nagent", "multi_agent_unguarded": "Multi-agent,\nunguarded",
         "multi_agent": "Multi-agent,\nbounded autonomy"}
COLOR = {"claude": "#4C72B0", "gpt": "#DD8452"}
MODEL = {"claude": "Claude Opus 5.5", "gpt": "GPT-5.5"}


def figure3():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ff = D["fixed_flow_s12"]["novel"]
    p = ff["success"] / ff["runs"]
    ax.bar(0, p, 0.55, color="#9E9E9E", yerr=[[p - ff["ci"][0]], [ff["ci"][1] - p]], capsize=3)
    ax.text(0, ff["ci"][1] + 0.03, f"{ff['success']}/{ff['runs']}", ha="center", fontsize=8)
    xt, xl = [0], ["Fixed\nflow"]
    for i, s in enumerate(ORDER):
        for j, m in enumerate(("claude", "gpt")):
            c = D["configs"][f"{m}|{s}"]["novel"]
            x = 1.4 + i * 1.3 + (j - 0.5) * 0.5
            p = c["success"] / c["runs"]
            ax.bar(x, p, 0.45, color=COLOR[m], yerr=[[p - c["ci"][0]], [c["ci"][1] - p]], capsize=3,
                   label=MODEL[m] if i == 0 else None)
            ax.text(x, c["ci"][1] + 0.03, f"{c['success']}/{c['runs']}", ha="center", fontsize=7.5)
        xt.append(1.4 + i * 1.3)
        xl.append(LABEL[s])
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Success on novel scenarios")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(HERE, "figures", "fig3_novel_success.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def figure4():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for j, m in enumerate(("claude", "gpt")):
        xs = [i + (j - 0.5) * 0.38 for i in range(3)]
        cs = [D["configs"][f"{m}|{s}"] for s in ORDER]
        a1.bar(xs, [c["touches"] / c["runs"] for c in cs], 0.36, color=COLOR[m], label=MODEL[m])
        a2.bar(xs, [c["cost_mean"] for c in cs], 0.36, color=COLOR[m], label=MODEL[m])
    ff = D["fixed_flow_s12"]
    a1.axhline(ff["touches"] / ff["runs"], color="#757575", ls="--", lw=1)
    a1.text(2.45, ff["touches"] / ff["runs"] + 0.03, "fixed flow", ha="right", fontsize=7.5, color="#555555")
    a1.set_ylabel("Human touches per change")
    a2.set_ylabel("Model cost per change (USD)")
    for a in (a1, a2):
        a.set_xticks(range(3))
        a.set_xticklabels([LABEL[s] for s in ORDER], fontsize=7.5)
        a.spines[["top", "right"]].set_visible(False)
    a2.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    path = os.path.join(HERE, "figures", "fig4_effort_cost.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path




if __name__ == "__main__":
    for f in (_fig1, _fig2, figure3, figure4):
        print(f())

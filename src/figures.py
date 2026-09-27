"""Figure style and layout checks.

Every figure has a single set of axes on a canvas 16.0 cm wide. Text is confined to the
margins: tick labels, axis labels and the legend must lie outside the axes rectangle, no
two text elements may overlap, nothing may extend beyond the canvas, and each axis must
carry at least two labeled ticks. `save` checks these conditions on the rendered figure and
raises if any fails, so a figure with a layout defect is never written. Captions belong to
the manuscript, so figures carry no titles.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

WIDTH_CM = 16.0
DPI = 600

# Okabe and Ito colour-blind safe palette
BLACK, ORANGE, SKY, GREEN = "#000000", "#E69F00", "#56B4E9", "#009E73"
YELLOW, BLUE, VERMILION, PURPLE = "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
CYCLE = [BLUE, VERMILION, GREEN, ORANGE, PURPLE, SKY, BLACK]


def setup():
    """Apply the common style."""
    mpl.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.constrained_layout.h_pad": 0.06,
        "figure.constrained_layout.w_pad": 0.06,
    })


def new(height_cm=8.5):
    """A single-panel figure of the standard width."""
    fig, ax = plt.subplots(figsize=(WIDTH_CM / 2.54, height_cm / 2.54), layout="constrained")
    return fig, ax


def legend(fig, where="right", ncol=1, order=None, **kw):
    """Figure legend placed outside the axes (right of them, or below them).

    `order` lists labels in the order they should appear; labels not listed follow.
    """
    loc = "outside right upper" if where == "right" else "outside lower center"
    handles, labels = [], []
    for ax in fig.axes:
        h, lab = ax.get_legend_handles_labels()
        handles += h
        labels += lab
    if order is not None:
        rank = {lab: i for i, lab in enumerate(order)}
        idx = sorted(range(len(labels)), key=lambda i: (rank.get(labels[i], len(order)), i))
        handles = [handles[i] for i in idx]
        labels = [labels[i] for i in idx]
    return fig.legend(handles, labels, loc=loc, ncol=ncol, frameon=False, **kw)


def fixed_ticks(ax, axis, values, labels=None):
    """Major ticks at the given values (with plain labels) and no minor tick labels."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(mpl.ticker.FixedLocator(values))
    a.set_major_formatter(mpl.ticker.FixedFormatter(labels or [f"{v:g}" for v in values]))
    a.set_minor_formatter(mpl.ticker.NullFormatter())


def sci(x, digits=2):
    """Number in scientific notation as mathtext, e.g. 3.4 x 10^-4."""
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10.0 ** e
    return rf"${m:.{digits - 1}f}\times10^{{{e}}}$"


def _text_boxes(fig, renderer):
    boxes = []
    for ax in fig.axes:
        for axis, name in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            lo, hi = sorted(axis.get_view_interval())
            span = hi - lo
            for which, ticks in (("major", axis.get_major_ticks()),
                                 ("minor", axis.get_minor_ticks())):
                for tk in ticks:
                    loc = tk.get_loc()
                    if not (lo - 1e-9 * abs(span) <= loc <= hi + 1e-9 * abs(span)):
                        continue
                    for lab in (tk.label1, tk.label2):
                        if lab.get_visible() and lab.get_text().strip():
                            boxes.append((f"{name} {which} tick label '{lab.get_text()}'",
                                          lab.get_window_extent(renderer)))
            if axis.label.get_visible() and axis.label.get_text().strip():
                boxes.append((f"{name} axis label", axis.label.get_window_extent(renderer)))
            off = axis.get_offset_text()
            if off.get_visible() and off.get_text().strip():
                boxes.append((f"{name} offset text", off.get_window_extent(renderer)))
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                boxes.append((f"text '{t.get_text()}'", t.get_window_extent(renderer)))
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            boxes.append(("axes legend", leg.get_window_extent(renderer)))
    for leg in fig.legends:
        if leg.get_visible():
            boxes.append(("figure legend", leg.get_window_extent(renderer)))
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            boxes.append((f"figure text '{t.get_text()}'", t.get_window_extent(renderer)))
    return boxes


def _overlap(a, b, tol=0.5):
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > tol) and (min(a.y1, b.y1) - max(a.y0, b.y0) > tol)


def _labeled_ticks(axis):
    """Number of ticks inside the view interval that carry a visible, non-empty label."""
    lo, hi = sorted(axis.get_view_interval())
    tol = 1e-9 * abs(hi - lo)
    n = 0
    for tk in axis.get_major_ticks() + axis.get_minor_ticks():
        if lo - tol <= tk.get_loc() <= hi + tol and any(
                lab.get_visible() and lab.get_text().strip() for lab in (tk.label1, tk.label2)):
            n += 1
    return n


def audit(fig):
    """Return a list of layout defects of a rendered figure (empty if the layout is clean)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []
    if len(fig.axes) != 1:
        problems.append(f"expected one axes, found {len(fig.axes)}")
    if fig.get_suptitle().strip():
        problems.append("figure has a suptitle")
    W, H = fig.bbox.width, fig.bbox.height
    boxes = _text_boxes(fig, renderer)
    for ax in fig.axes:
        if ax.get_title().strip():
            problems.append("axes has a title")
        for axis, name in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            if _labeled_ticks(axis) < 2:
                problems.append(f"{name} axis has fewer than two labeled ticks")
        frame = ax.get_window_extent(renderer)
        for name, bb in boxes:
            if _overlap(bb, frame):
                problems.append(f"{name} intrudes into the axes")
    for name, bb in boxes:
        if bb.x0 < 1.0 or bb.y0 < 1.0 or bb.x1 > W - 1.0 or bb.y1 > H - 1.0:
            problems.append(f"{name} extends beyond the canvas")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _overlap(boxes[i][1], boxes[j][1]):
                problems.append(f"{boxes[i][0]} overlaps {boxes[j][0]}")
    return problems


def save(fig, name, outdir="figures"):
    """Audit the layout, then write <name>.png (600 dpi) and <name>.pdf."""
    problems = audit(fig)
    if problems:
        raise AssertionError(f"{name}: " + "; ".join(problems))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=DPI)
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None})


def shade(ax, lo, hi, color="0.88", **kw):
    """Vertical band marking a frequency interval (no text is drawn)."""
    return ax.axvspan(lo, hi, color=color, lw=0, zorder=0, **kw)


def log_ticks(ax, axis="y"):
    """Decade ticks with mathtext labels and unlabeled minor ticks."""
    loc = mpl.ticker.LogLocator(base=10.0, numticks=12)
    minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12)
    fmt = mpl.ticker.LogFormatterMathtext()
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(loc)
    a.set_minor_locator(minor)
    a.set_major_formatter(fmt)
    a.set_minor_formatter(mpl.ticker.NullFormatter())

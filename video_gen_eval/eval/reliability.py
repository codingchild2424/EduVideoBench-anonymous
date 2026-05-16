"""Inter-rater reliability and bias analysis for dual VLM judges.

Computes agreement metrics between VLM judges (e.g. Gemini 3 Pro and
GPT-4o) to validate evaluation objectivity.  Provides:

* **Cohen's weighted kappa** (κ_w) for ordinal agreement on binned scores.
* **ICC(2,1)** (two-way random, single measures) for continuous score agreement.
* **Pearson correlation** between judge score vectors.
* **Bias analysis**: per-model paired differences to detect manufacturer
  bias (e.g. Gemini scoring Veo3 higher than GPT-4o does).

All statistics are computed without external dependencies (no scipy/sklearn).
"""

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .models import ItemEvalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_paired_scores(
    all_judge_results: Dict[str, List[ItemEvalResult]],
) -> Dict[str, Dict[str, float]]:
    """Organise per-judge results into paired score vectors.

    Args:
        all_judge_results: Mapping of judge_id to list of ItemEvalResults.
            The pipeline should collect these *before* merging.

    Returns:
        Dict mapping prompt_id to {judge_id: normalised_score}.
        Only prompt_ids scored by **all** judges are included.
    """
    judges = list(all_judge_results.keys())
    if len(judges) < 2:
        return {}

    # Build prompt_id -> judge_id -> score
    paired: Dict[str, Dict[str, float]] = defaultdict(dict)
    for judge_id, results in all_judge_results.items():
        for r in results:
            if r.error is None:
                paired[r.prompt_id][judge_id] = r.score

    # Keep only prompt_ids scored by all judges
    complete = {
        pid: scores
        for pid, scores in paired.items()
        if len(scores) == len(judges)
    }

    logger.info(
        "Collected %d paired scores across %d judges (of %d total prompts)",
        len(complete), len(judges),
        max(len(r) for r in all_judge_results.values()),
    )
    return complete


# ---------------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------------

def compute_pearson_r(
    x: List[float], y: List[float]
) -> Tuple[float, float]:
    """Pearson correlation coefficient and approximate two-tailed p-value.

    Uses the t-distribution approximation for the p-value:
        t = r * sqrt(n - 2) / sqrt(1 - r^2)
    with n - 2 degrees of freedom.

    Args:
        x: First score vector.
        y: Second score vector (same length).

    Returns:
        Tuple of (r, p_value). Returns (0.0, 1.0) if computation fails.
    """
    n = len(x)
    if n < 3 or len(y) != n:
        return 0.0, 1.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0, 1.0

    r = cov / denom

    # Clamp for numerical safety
    r = max(-1.0, min(1.0, r))

    # t-test approximation for p-value
    if abs(r) >= 1.0:
        p_value = 0.0
    else:
        t_stat = r * math.sqrt((n - 2) / (1.0 - r * r))
        p_value = _t_distribution_2tail_p(t_stat, n - 2)

    return r, p_value


def _t_distribution_2tail_p(t: float, df: int) -> float:
    """Approximate two-tailed p-value for a t-statistic.

    Uses the regularised incomplete beta function relationship.
    For large df, falls back to normal approximation.
    """
    if df <= 0:
        return 1.0

    # Normal approximation for large df
    if df > 100:
        # Use standard normal tail probability
        z = abs(t)
        return 2.0 * _normal_sf(z)

    # Beta distribution approximation: p = I_{df/(df+t^2)}(df/2, 1/2)
    x = df / (df + t * t)
    p = _regularised_beta(x, df / 2.0, 0.5)
    return max(0.0, min(1.0, p))


def _normal_sf(z: float) -> float:
    """Standard normal survival function P(Z > z)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _regularised_beta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta function I_x(a, b) via continued fraction.

    Uses Lentz's algorithm for the continued fraction expansion.
    Sufficient for our purpose (moderate a, b values).
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use the symmetry relation if needed for convergence
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _regularised_beta(1.0 - x, b, a)

    # Log-beta for the prefactor
    ln_prefactor = (
        a * math.log(x)
        + b * math.log(1.0 - x)
        - math.log(a)
        - _log_beta(a, b)
    )

    try:
        prefactor = math.exp(ln_prefactor)
    except OverflowError:
        return 0.5  # fallback

    # Continued fraction (Lentz's method)
    tiny = 1e-30
    f = tiny
    c = tiny
    d = 0.0

    for m in range(1, 200):
        # Even step: d_{2m}
        m2 = 2 * m
        num = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        f *= c * d

        # Odd step: d_{2m+1}
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < 1e-10:
            break

    return prefactor * f


def _log_beta(a: float, b: float) -> float:
    """Log of the beta function B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# ---------------------------------------------------------------------------
# Cohen's weighted kappa
# ---------------------------------------------------------------------------

_N_BINS = 5  # Ordinal bins for kappa: [0, 0.2), [0.2, 0.4), ..., [0.8, 1.0]


def _score_to_bin(score: float, n_bins: int = _N_BINS) -> int:
    """Convert a [0, 1] score to an ordinal bin index."""
    b = int(score * n_bins)
    return min(b, n_bins - 1)


def compute_cohens_kappa(
    scores_a: List[float],
    scores_b: List[float],
    n_bins: int = _N_BINS,
    weight_type: str = "quadratic",
) -> float:
    """Cohen's weighted kappa for ordinal agreement.

    Args:
        scores_a: Normalised scores from judge A.
        scores_b: Normalised scores from judge B (same length).
        n_bins: Number of ordinal bins.
        weight_type: ``"linear"`` or ``"quadratic"`` weighting.

    Returns:
        Weighted kappa in [-1, 1]. Returns 0.0 if computation fails.
    """
    n = len(scores_a)
    if n == 0 or len(scores_b) != n:
        return 0.0

    bins_a = [_score_to_bin(s, n_bins) for s in scores_a]
    bins_b = [_score_to_bin(s, n_bins) for s in scores_b]

    # Build observed agreement matrix
    observed = [[0] * n_bins for _ in range(n_bins)]
    for a, b in zip(bins_a, bins_b):
        observed[a][b] += 1

    # Marginals
    row_sums = [sum(observed[i]) for i in range(n_bins)]
    col_sums = [sum(observed[i][j] for i in range(n_bins)) for j in range(n_bins)]

    # Weight matrix
    weights = [[0.0] * n_bins for _ in range(n_bins)]
    for i in range(n_bins):
        for j in range(n_bins):
            if weight_type == "linear":
                weights[i][j] = abs(i - j) / (n_bins - 1) if n_bins > 1 else 0.0
            else:  # quadratic
                weights[i][j] = ((i - j) ** 2) / ((n_bins - 1) ** 2) if n_bins > 1 else 0.0

    # Weighted observed disagreement
    po = sum(
        weights[i][j] * observed[i][j]
        for i in range(n_bins)
        for j in range(n_bins)
    ) / n

    # Weighted expected disagreement
    pe = sum(
        weights[i][j] * row_sums[i] * col_sums[j]
        for i in range(n_bins)
        for j in range(n_bins)
    ) / (n * n)

    if pe == 0:
        return 1.0  # Perfect agreement

    kappa = 1.0 - po / pe
    return kappa


# ---------------------------------------------------------------------------
# ICC(2,1) — Two-way random, single measures
# ---------------------------------------------------------------------------

def compute_icc(
    paired_scores: Dict[str, Dict[str, float]],
) -> Tuple[float, Dict[str, Any]]:
    """Intraclass Correlation Coefficient ICC(2,1).

    Two-way random effects, absolute agreement, single measures.
    This is the appropriate ICC type for assessing agreement between
    two judges who represent a random sample of possible judges.

    Formula:
        ICC(2,1) = (MS_R - MS_E) / (MS_R + (k-1)*MS_E + k*(MS_C - MS_E)/n)

    where:
        MS_R = between-subjects (prompts) mean square
        MS_C = between-raters (judges) mean square
        MS_E = residual mean square
        n = number of subjects (prompts)
        k = number of raters (judges)

    Args:
        paired_scores: Dict mapping prompt_id to {judge_id: score}.

    Returns:
        Tuple of (icc_value, anova_details).
    """
    if not paired_scores:
        return 0.0, {"error": "no paired scores"}

    prompt_ids = sorted(paired_scores.keys())
    judge_ids = sorted(next(iter(paired_scores.values())).keys())

    n = len(prompt_ids)  # subjects
    k = len(judge_ids)   # raters

    if n < 2 or k < 2:
        return 0.0, {"error": f"need at least 2 subjects and 2 raters, got n={n}, k={k}"}

    # Build score matrix: n x k
    matrix = []
    for pid in prompt_ids:
        row = [paired_scores[pid][jid] for jid in judge_ids]
        matrix.append(row)

    # Grand mean
    grand_mean = sum(s for row in matrix for s in row) / (n * k)

    # Row means (subject means)
    row_means = [sum(row) / k for row in matrix]

    # Column means (rater means)
    col_means = [
        sum(matrix[i][j] for i in range(n)) / n for j in range(k)
    ]

    # Sum of squares
    ss_total = sum(
        (matrix[i][j] - grand_mean) ** 2
        for i in range(n) for j in range(k)
    )

    ss_rows = k * sum((rm - grand_mean) ** 2 for rm in row_means)
    ss_cols = n * sum((cm - grand_mean) ** 2 for cm in col_means)
    ss_error = ss_total - ss_rows - ss_cols

    # Mean squares
    df_rows = n - 1
    df_cols = k - 1
    df_error = (n - 1) * (k - 1)

    ms_r = ss_rows / df_rows if df_rows > 0 else 0.0
    ms_c = ss_cols / df_cols if df_cols > 0 else 0.0
    ms_e = ss_error / df_error if df_error > 0 else 0.0

    # ICC(2,1) formula
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    if denom <= 0:
        icc = 0.0
    else:
        icc = (ms_r - ms_e) / denom

    # Clamp to [-1, 1]
    icc = max(-1.0, min(1.0, icc))

    details = {
        "n_subjects": n,
        "k_raters": k,
        "grand_mean": round(grand_mean, 6),
        "ms_rows": round(ms_r, 6),
        "ms_cols": round(ms_c, 6),
        "ms_error": round(ms_e, 6),
        "ss_rows": round(ss_rows, 6),
        "ss_cols": round(ss_cols, 6),
        "ss_error": round(ss_error, 6),
    }

    return round(icc, 6), details


# ---------------------------------------------------------------------------
# Bias analysis
# ---------------------------------------------------------------------------

def compute_bias_analysis(
    paired_scores: Dict[str, Dict[str, float]],
    prompt_model_map: Dict[str, str],
    known_affiliations: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Detect systematic scoring bias between judges, especially manufacturer bias.

    For each pair of judges, computes:
    - Overall mean difference (signed) and its significance.
    - Per-video-model mean difference to detect manufacturer bias
      (e.g. Gemini scoring Veo3 higher than GPT-4o does).

    Uses Wilcoxon-style signed-rank approximation for significance.

    Args:
        paired_scores: prompt_id -> {judge_id: score}.
        prompt_model_map: prompt_id -> video_model_id (which model generated
            the video being scored).
        known_affiliations: Optional mapping of judge_id to list of
            affiliated video model IDs (e.g. ``{"gemini-3-pro": ["veo31"]}``).
            Used to flag potential manufacturer bias.

    Returns:
        Dictionary with bias analysis results.
    """
    if not paired_scores:
        return {"error": "no paired scores"}

    judge_ids = sorted(next(iter(paired_scores.values())).keys())
    if len(judge_ids) < 2:
        return {"error": "need at least 2 judges"}

    if known_affiliations is None:
        known_affiliations = {
            "gemini-3-pro": ["veo31"],
            "gpt-4o": ["sora2"],
        }

    results: Dict[str, Any] = {
        "judge_ids": judge_ids,
        "pairwise_comparisons": [],
    }

    # Pairwise comparison for all judge pairs
    for i in range(len(judge_ids)):
        for j in range(i + 1, len(judge_ids)):
            ja, jb = judge_ids[i], judge_ids[j]
            pair_result = _analyse_judge_pair(
                ja, jb, paired_scores, prompt_model_map, known_affiliations,
            )
            results["pairwise_comparisons"].append(pair_result)

    return results


def _analyse_judge_pair(
    judge_a: str,
    judge_b: str,
    paired_scores: Dict[str, Dict[str, float]],
    prompt_model_map: Dict[str, str],
    affiliations: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Analyse scoring differences between a pair of judges."""
    # Collect paired differences: d = score_a - score_b
    diffs: List[float] = []
    by_model: Dict[str, List[float]] = defaultdict(list)

    for pid, scores in paired_scores.items():
        sa = scores.get(judge_a)
        sb = scores.get(judge_b)
        if sa is not None and sb is not None:
            d = sa - sb
            diffs.append(d)
            model_id = prompt_model_map.get(pid, "unknown")
            by_model[model_id].append(d)

    n = len(diffs)
    if n == 0:
        return {
            "judges": [judge_a, judge_b],
            "error": "no paired data",
        }

    mean_diff = sum(diffs) / n
    abs_mean_diff = sum(abs(d) for d in diffs) / n
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / n) if n > 1 else 0.0

    # Signed-rank test approximation (Wilcoxon)
    p_value = _wilcoxon_p_approx(diffs)

    # Per video-model breakdown
    model_bias: Dict[str, Dict[str, Any]] = {}
    for model_id, model_diffs in sorted(by_model.items()):
        m = len(model_diffs)
        m_mean = sum(model_diffs) / m
        m_std = math.sqrt(sum((d - m_mean) ** 2 for d in model_diffs) / m) if m > 1 else 0.0

        # Check if this model is affiliated with either judge
        affiliated_judge = None
        for jid, affiliated_models in affiliations.items():
            if model_id in affiliated_models:
                affiliated_judge = jid

        model_bias[model_id] = {
            "n": m,
            "mean_diff": round(m_mean, 6),
            "std_diff": round(m_std, 6),
            "affiliated_judge": affiliated_judge,
            "bias_direction": (
                f"{judge_a} scores higher"
                if m_mean > 0.01
                else f"{judge_b} scores higher"
                if m_mean < -0.01
                else "no significant bias"
            ),
        }

    # Flag manufacturer bias
    manufacturer_bias_flags = []
    for model_id, info in model_bias.items():
        if info["affiliated_judge"] is not None:
            aff_judge = info["affiliated_judge"]
            m_diff = info["mean_diff"]
            # Positive mean_diff means judge_a scores higher
            # If affiliated judge is judge_a and scores higher → bias
            if aff_judge == judge_a and m_diff > 0.05:
                manufacturer_bias_flags.append({
                    "judge": aff_judge,
                    "video_model": model_id,
                    "mean_diff": m_diff,
                    "severity": "high" if m_diff > 0.10 else "moderate",
                })
            elif aff_judge == judge_b and m_diff < -0.05:
                manufacturer_bias_flags.append({
                    "judge": aff_judge,
                    "video_model": model_id,
                    "mean_diff": abs(m_diff),
                    "severity": "high" if abs(m_diff) > 0.10 else "moderate",
                })

    return {
        "judges": [judge_a, judge_b],
        "n_pairs": n,
        "overall": {
            "mean_diff": round(mean_diff, 6),
            "abs_mean_diff": round(abs_mean_diff, 6),
            "std_diff": round(std_diff, 6),
            "p_value": round(p_value, 6),
            "significant_at_005": p_value < 0.05,
        },
        "per_video_model": model_bias,
        "manufacturer_bias_flags": manufacturer_bias_flags,
    }


def _wilcoxon_p_approx(diffs: List[float]) -> float:
    """Approximate Wilcoxon signed-rank test p-value (two-tailed).

    Normal approximation with continuity correction for n > 20.
    For small n, returns a conservative estimate.
    """
    # Remove zeros
    nonzero = [d for d in diffs if abs(d) > 1e-10]
    n = len(nonzero)

    if n < 5:
        return 1.0  # too few observations

    # Rank absolute values
    abs_vals = [(abs(d), i) for i, d in enumerate(nonzero)]
    abs_vals.sort()

    # Assign ranks (average ties)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs(abs_vals[j][0] - abs_vals[i][0]) < 1e-10:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average rank
        for k in range(i, j):
            ranks[abs_vals[k][1]] = avg_rank
        i = j

    # T+ = sum of ranks for positive differences
    t_plus = sum(ranks[i] for i in range(n) if nonzero[i] > 0)

    # Expected value and variance under H0
    expected = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0

    if var <= 0:
        return 1.0

    # Normal approximation with continuity correction
    z = (abs(t_plus - expected) - 0.5) / math.sqrt(var)
    p = 2.0 * _normal_sf(z)

    return min(1.0, p)


# ---------------------------------------------------------------------------
# Comprehensive reliability report
# ---------------------------------------------------------------------------

def compute_reliability_metrics(
    all_judge_results: Dict[str, List[ItemEvalResult]],
    prompt_model_map: Optional[Dict[str, str]] = None,
    known_affiliations: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Compute all inter-rater reliability metrics.

    This is the main entry point for the reliability module. Call after
    scoring all prompts with all judges but before merging.

    Args:
        all_judge_results: judge_id -> list of ItemEvalResults.
        prompt_model_map: prompt_id -> video model that generated the video.
        known_affiliations: judge_id -> affiliated video model IDs.

    Returns:
        Dictionary containing all reliability metrics.
    """
    paired = collect_paired_scores(all_judge_results)

    if not paired:
        return {
            "status": "skipped",
            "reason": "insufficient paired data",
            "n_judges": len(all_judge_results),
        }

    judge_ids = sorted(next(iter(paired.values())).keys())
    n_pairs = len(paired)

    # Extract score vectors for each judge pair
    report: Dict[str, Any] = {
        "status": "computed",
        "n_paired_items": n_pairs,
        "judge_ids": judge_ids,
    }

    # For two judges, compute directly
    if len(judge_ids) == 2:
        scores_a = [paired[pid][judge_ids[0]] for pid in sorted(paired)]
        scores_b = [paired[pid][judge_ids[1]] for pid in sorted(paired)]

        # Pearson r
        r, r_p = compute_pearson_r(scores_a, scores_b)
        report["pearson"] = {
            "r": round(r, 6),
            "p_value": round(r_p, 6),
            "significant_at_005": r_p < 0.05,
            "interpretation": _interpret_correlation(r),
        }

        # Cohen's kappa
        kappa = compute_cohens_kappa(scores_a, scores_b)
        report["cohens_kappa"] = {
            "kappa_weighted": round(kappa, 6),
            "weight_type": "quadratic",
            "interpretation": _interpret_kappa(kappa),
        }

    # ICC (works for 2+ judges)
    icc_val, icc_details = compute_icc(paired)
    report["icc"] = {
        "icc_2_1": round(icc_val, 6),
        "interpretation": _interpret_icc(icc_val),
        "anova_details": icc_details,
    }

    # Bias analysis
    if prompt_model_map:
        bias = compute_bias_analysis(
            paired, prompt_model_map, known_affiliations
        )
        report["bias_analysis"] = bias
    else:
        report["bias_analysis"] = {
            "status": "skipped",
            "reason": "no prompt_model_map provided",
        }

    # Per-category reliability
    report["per_category"] = _compute_per_category_reliability(
        all_judge_results
    )

    return report


def _compute_per_category_reliability(
    all_judge_results: Dict[str, List[ItemEvalResult]],
) -> Dict[str, Any]:
    """Compute reliability metrics broken down by KSA category."""
    judge_ids = sorted(all_judge_results.keys())
    if len(judge_ids) < 2:
        return {}

    # Group results by category for each judge
    by_category_judge: Dict[str, Dict[str, List[ItemEvalResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for judge_id, results in all_judge_results.items():
        for r in results:
            category = r.details.get("category", "UNKNOWN")
            by_category_judge[category][judge_id].append(r)

    cat_reliability: Dict[str, Any] = {}
    for category in sorted(by_category_judge):
        cat_judge_results = by_category_judge[category]
        paired = collect_paired_scores(cat_judge_results)

        if len(paired) < 3:
            cat_reliability[category] = {
                "n_pairs": len(paired),
                "status": "insufficient_data",
            }
            continue

        scores_a = [paired[pid][judge_ids[0]] for pid in sorted(paired)]
        scores_b = [paired[pid][judge_ids[1]] for pid in sorted(paired)]

        r, _ = compute_pearson_r(scores_a, scores_b)
        kappa = compute_cohens_kappa(scores_a, scores_b)

        cat_reliability[category] = {
            "n_pairs": len(paired),
            "pearson_r": round(r, 6),
            "kappa_weighted": round(kappa, 6),
        }

    return cat_reliability


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------

def _interpret_correlation(r: float) -> str:
    """Interpret Pearson correlation strength."""
    ar = abs(r)
    if ar >= 0.90:
        return "very strong"
    if ar >= 0.70:
        return "strong"
    if ar >= 0.50:
        return "moderate"
    if ar >= 0.30:
        return "weak"
    return "negligible"


def _interpret_kappa(kappa: float) -> str:
    """Interpret Cohen's kappa using Landis & Koch (1977) scale."""
    if kappa >= 0.81:
        return "almost perfect"
    if kappa >= 0.61:
        return "substantial"
    if kappa >= 0.41:
        return "moderate"
    if kappa >= 0.21:
        return "fair"
    if kappa >= 0.0:
        return "slight"
    return "poor"


def _interpret_icc(icc: float) -> str:
    """Interpret ICC using Koo & Li (2016) scale."""
    if icc >= 0.90:
        return "excellent"
    if icc >= 0.75:
        return "good"
    if icc >= 0.50:
        return "moderate"
    return "poor"


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_reliability_report(metrics: Dict[str, Any]) -> str:
    """Generate a Markdown reliability report from computed metrics.

    Args:
        metrics: Output of :func:`compute_reliability_metrics`.

    Returns:
        Markdown-formatted string.
    """
    lines: List[str] = []
    lines.append("# Inter-Rater Reliability Report")
    lines.append("")

    if metrics.get("status") != "computed":
        lines.append(f"**Status**: {metrics.get('status', 'unknown')}")
        lines.append(f"**Reason**: {metrics.get('reason', 'unknown')}")
        return "\n".join(lines)

    lines.append(f"**Judges**: {', '.join(metrics.get('judge_ids', []))}")
    lines.append(f"**Paired items**: {metrics.get('n_paired_items', 0)}")
    lines.append("")

    # Agreement summary table
    lines.append("## Agreement Metrics")
    lines.append("")
    lines.append("| Metric | Value | Interpretation |")
    lines.append("|--------|-------|----------------|")

    if "pearson" in metrics:
        p = metrics["pearson"]
        lines.append(
            f"| Pearson r | {p['r']:.4f} (p={p['p_value']:.4f}) | "
            f"{p['interpretation']} |"
        )

    if "cohens_kappa" in metrics:
        k = metrics["cohens_kappa"]
        lines.append(
            f"| Cohen's kappa (weighted) | {k['kappa_weighted']:.4f} | "
            f"{k['interpretation']} |"
        )

    if "icc" in metrics:
        i = metrics["icc"]
        lines.append(
            f"| ICC(2,1) | {i['icc_2_1']:.4f} | "
            f"{i['interpretation']} |"
        )

    lines.append("")

    # Bias analysis
    bias = metrics.get("bias_analysis", {})
    if bias.get("pairwise_comparisons"):
        lines.append("## Bias Analysis")
        lines.append("")

        for comp in bias["pairwise_comparisons"]:
            judges = comp.get("judges", [])
            overall = comp.get("overall", {})
            lines.append(
                f"### {judges[0]} vs {judges[1]} "
                f"(n={comp.get('n_pairs', 0)})"
            )
            lines.append("")
            lines.append(
                f"- Mean difference ({judges[0]} - {judges[1]}): "
                f"{overall.get('mean_diff', 0):.4f} "
                f"(p={overall.get('p_value', 1.0):.4f})"
            )
            lines.append(
                f"- Absolute mean difference: "
                f"{overall.get('abs_mean_diff', 0):.4f}"
            )
            lines.append("")

            # Per video-model
            per_model = comp.get("per_video_model", {})
            if per_model:
                lines.append("| Video Model | n | Mean Diff | Affiliated Judge |")
                lines.append("|-------------|---|-----------|------------------|")
                for model_id, info in sorted(per_model.items()):
                    aff = info.get("affiliated_judge", "-")
                    lines.append(
                        f"| {model_id} | {info['n']} | "
                        f"{info['mean_diff']:+.4f} | {aff or '-'} |"
                    )
                lines.append("")

            # Manufacturer bias flags
            flags = comp.get("manufacturer_bias_flags", [])
            if flags:
                lines.append("**Manufacturer Bias Flags:**")
                for flag in flags:
                    lines.append(
                        f"- {flag['judge']} favours {flag['video_model']} "
                        f"(diff={flag['mean_diff']:.4f}, "
                        f"severity={flag['severity']})"
                    )
                lines.append("")
            else:
                lines.append("No manufacturer bias detected.")
                lines.append("")

    # Per-category reliability
    per_cat = metrics.get("per_category", {})
    if per_cat:
        lines.append("## Per-Category Reliability")
        lines.append("")
        lines.append("| Category | Pairs | Pearson r | Kappa |")
        lines.append("|----------|-------|-----------|-------|")
        for cat, info in sorted(per_cat.items()):
            if info.get("status") == "insufficient_data":
                lines.append(f"| {cat} | {info.get('n_pairs', 0)} | - | - |")
            else:
                lines.append(
                    f"| {cat} | {info['n_pairs']} | "
                    f"{info.get('pearson_r', 0):.4f} | "
                    f"{info.get('kappa_weighted', 0):.4f} |"
                )
        lines.append("")

    lines.append("---")
    lines.append("*Generated by EduVBench Reliability Analysis Module*")
    lines.append("")
    return "\n".join(lines)

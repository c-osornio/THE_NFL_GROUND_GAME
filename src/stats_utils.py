from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency


def chi2_with_cramers_v(table: pd.DataFrame) -> dict:
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape
    v = np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else np.nan
    return {"chi2": chi2, "p_value": p, "dof": dof, "cramers_v": v, "expected": expected}


def poisson_rate_ratio(count1: int, exp1: float, count2: int, exp2: float, alpha: float = 0.05) -> dict:
    # IRR = (count1/exp1) / (count2/exp2)
    rate1 = count1 / exp1
    rate2 = count2 / exp2
    irr = rate1 / rate2
    # Wald CI on log scale
    se = np.sqrt(1.0 / count1 + 1.0 / count2)
    z = 1.959963984540054
    lcl = np.exp(np.log(irr) - z * se)
    ucl = np.exp(np.log(irr) + z * se)
    return {"irr": irr, "lcl": lcl, "ucl": ucl}


def fit_poisson_glm(
    df: pd.DataFrame,
    formula: str,
    offset_col: Optional[str] = None,
    cluster_col: Optional[str] = None,
):
    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson())
    if offset_col is not None and offset_col in df.columns:
        model = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=np.log(df[offset_col]))
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]}) if cluster_col else model.fit()
    return result


def benjamini_hochberg(pvals: Iterable[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    pvals = np.asarray(list(pvals))
    n = len(pvals)
    order = np.argsort(pvals)
    ordered = pvals[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    significant = ordered <= thresholds
    max_k = significant.nonzero()[0].max() + 1 if significant.any() else 0
    cutoff = thresholds[max_k - 1] if max_k > 0 else 0
    return (pvals <= cutoff), pvals
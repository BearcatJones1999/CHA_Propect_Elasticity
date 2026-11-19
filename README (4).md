# CHA Labor-to-Volume Elasticity

Estimates how much worked hours change when volume changes, by PROSPECT department, using CHA benchmarking data across children's hospitals. It also tests whether Cincinnati (CIN) units flex labor differently from a peer set of top-ranked children's hospitals.

The question it answers: **if a department's volume goes up 10%, how much do its worked hours go up?**

---

## Quick start

```bash
pip install pandas numpy statsmodels openpyxl pyarrow
python cha_elasticity.py --input "path/to/Prospect Data Clean.xlsx" --output cha_elasticity_results.xlsx
```

With no arguments it uses the default network path set in the config block at the top of the script. The first run on the Excel file writes a `.parquet` cache next to it, so later runs load in seconds.

Settings at the top of the script:

| Setting | Purpose |
|---|---|
| `PEER_PREFIXES` | Hospital code prefixes that define the peer set |
| `HOME_PREFIX` | Prefix for Cincinnati units (`CIN-`) |
| `TARGET_DEPTS` | PROSPECT departments to model (`None` = all) |
| `MIN_OBS`, `MIN_CLUSTERS` | Minimum data needed before a department is modeled |
| `SCENARIO_CHANGE` | Volume change used to translate elasticities into hours (default 10%) |

---

## What the pipeline does

1. **Load.** Reads the CHA extract (job-title level, around 537k rows) and keeps only the seven columns needed.
2. **Collapse to a panel.** Volume and worked hours are reported at the cost-center level and repeated on every job-title row, so the data is collapsed to one row per **hospital × cost center (CCRef) × quarter**. That combination is the unit of observation.
3. **Clean.** Keeps rows with positive volume and hours (required for logs), and flags CIN and peer-set hospitals.
4. **Estimate.** Runs three analyses, each under two specifications (described below).
5. **Export.** Writes one Excel workbook with three sheets: `pooled`, `by_department` and `cin_vs_peer`.

---

## The math

### The model

ln(Hours) = β · ln(Volume) + fixed effects + ε

Because both sides are in logs, **β is an elasticity**: a 1% change in volume goes with a β% change in worked hours.

| β | Meaning | What happens to hours per unit (HPU) as volume grows |
|---|---|---|
| β < 1 | **Economies of scale.** Part of staffing is fixed (charge nurse, minimum staffing, leadership) | HPU falls |
| β = 1 | **Proportional.** Labor is fully variable | HPU flat |
| β > 1 | **Diseconomies.** Each extra unit costs more labor (overtime, agency, acuity) | HPU rises |

A useful identity: the elasticity of HPU with respect to volume is **β − 1**.

Turning β into a planning number for a volume change of g:

Hours_new = Hours_old × (1 + g)^β

Example: β = 0.57 and volume +20% gives hours × 1.2^0.57 ≈ **+11%**, so HPU drops about 7.5% purely from scale.

### Two specifications

Every model is run two ways because they answer different questions.

| Spec | Fixed effects | Variation used | Question answered |
|---|---|---|---|
| `cross_section` | Department + quarter | Bigger units vs. smaller units | Do larger departments structurally need fewer hours per unit? |
| `within_unit` | Cost center + quarter | Each unit vs. itself over time | When *this* unit's volume moves, how much do *its* hours move? |

- **Quarter fixed effects** remove shocks that hit every hospital in the same quarter (seasonality, industry staffing trends).
- **Cost-center fixed effects** remove everything stable about a unit: its size, layout, acuity mix and local labor market. What's left is the unit's own quarter-to-quarter flexing.

For budgeting and flex-staffing decisions, `within_unit` is usually the more relevant number. `cross_section` is better for understanding structural scale (for example, the case for consolidating small units).

Fixed effects are absorbed by iterative demeaning instead of creating thousands of dummy columns. That gives identical coefficients to the dummy-variable regression, but runs much faster and uses far less memory.

### Standard errors

The same cost center appears every quarter, so its errors are correlated over time. Standard errors are **clustered by cost center**. Plain robust (HC1) errors would treat each quarter as independent and overstate precision.

### Scale classification

A department is labeled from its 95% confidence interval, not the point estimate:
- CI entirely below 1 → *Economies of scale*
- CI entirely above 1 → *Diseconomies of scale*
- CI includes 1 → *Proportional (CI includes 1)*

### Cincinnati vs. peers

Within each department:

ln(Hours) = β · ln(Volume) + c · CIN + γ · (ln(Volume) × CIN) + fixed effects + ε

- **β** = peer elasticity
- **β + γ** = Cincinnati elasticity
- **γ** = difference in slope; the test of interest is γ = 0
- **c** lets Cincinnati have its own staffing *level*, so γ picks up only a difference in *responsiveness*. Without c, a level gap would be pushed into γ and bias it. In the `within_unit` spec, c is absorbed by the cost-center fixed effects.

Reading the verdict:
- **γ < 0:** CIN hours are *less* volume-sensitive than peers'. Growth is absorbed more efficiently, but hours also come down less when volume falls.
- **γ > 0:** CIN hours are *more* volume-sensitive. CIN adds more labor per unit of growth than peers do.

To judge materiality, `cin_excess_hours_per_qtr_if_vol_+10%` converts γ into hours. It shows how many more (or fewer) worked hours an average CIN unit in that department would add per quarter after a 10% volume increase, compared with peer behavior. A gap can be statistically significant and still too small to act on. This column tells you which case you're in.

---

## Output columns

**`pooled`** and **`by_department`**

| Column | Meaning |
|---|---|
| `spec` | `cross_section` or `within_unit` |
| `n_obs`, `n_units` | Unit-quarters used, and distinct cost centers (clusters) |
| `elasticity` | β |
| `se_clustered`, `ci_low`, `ci_high` | Clustered SE and 95% CI |
| `scale_behavior` | CI-based classification |
| `hours_chg_if_vol_+10%` | Implied % change in hours for a 10% volume increase |
| `hpu_chg_if_vol_+10%` | Implied % change in hours per unit |
| `r2_within` | R² after removing fixed effects (how much hour variation volume explains) |

**`cin_vs_peer`**

| Column | Meaning |
|---|---|
| `peer_elasticity`, `cin_elasticity` | β and β + γ |
| `diff_cin_minus_peer` | γ, with `se_clustered`, `p_value`, `ci_low`, `ci_high` |
| `verdict` | Significance and direction at α = 0.05 |
| `cin_excess_hours_per_qtr_if_vol_+10%` | Materiality of the gap, in hours |
| `n_cin_units`, `low_cin_units_warning` | Flags results that rest on fewer than 3 CIN cost centers |

---

## Why this is useful

1. **Volume-driven budgeting.** Linear budgets assume hours scale 1:1 with volume. For a department with β = 0.85, a linear budget overstates the labor needed for growth and understates the reduction needed when volume falls. Use Hours₀ × (V₁/V₀)^β for FY budget and forecast scenarios.
2. **Fairer benchmarking.** Raw HPU percentiles penalize small units whenever β < 1, because fixed staffing gets spread over less volume. A unit can look inefficient just from its size. The elasticity separates scale effects from real productivity gaps before anything is called an "opportunity."
3. **Marginal cost of growth.** Low-β departments can absorb new volume cheaply, and high-β departments need labor added almost one-for-one. That feeds into service-line growth and capacity decisions.
4. **Flex-staffing diagnostics.** The CIN vs. peer comparison shows which departments flex labor differently from top-ranked peers. That points to where scheduling, float pools or core staffing levels deserve a closer look.
5. **Consolidation decisions.** A strong cross-sectional scale economy (low `cross_section` β) is evidence that combining small units could lower HPU.

---

## Limitations

- **Association, not causation.** Volume isn't randomly assigned. If acuity rises together with volume, β will pick up some of that. The within-unit spec controls for stable acuity differences between units, but not for acuity changes over time.
- **Volume definitions differ by department.** Units of service (patient days, visits, tests) differ across PROSPECT departments, so elasticities are comparable *across* departments as percentages, but raw volumes are not.
- **Small CIN samples.** Some departments have only a few CIN cost centers, which makes γ noisy. Check `n_cin_units`.
- **Short panels.** With few quarters per unit, `within_unit` estimates rely on limited within-unit variation. When the two specs disagree sharply, the gap is information in itself (structural scale vs. short-run flexing), not just an error.
- **Worked hours only.** Paid hours, skill mix and agency use aren't modeled. Adding them would show *how* departments flex (for example, through overtime vs. core staff).

---

## Changes from the original notebook

- All plots removed. The output is tables only.
- Added the missing CIN main effect to the comparison model, so the slope difference isn't biased by staffing-level differences.
- Standard errors are clustered by cost center instead of HC1.
- Added the `within_unit` specification (cost-center fixed effects).
- Scale classification uses confidence intervals, not fixed 0.95/1.05 point cutoffs.
- Added hours-based materiality columns.
- One script, one config block, one Excel output.

# Calendar-Horizon Semantics Diagnosis

## 1. Problem statement

With H2 target configuration (`horizon_years: 2`, `include_publication_year: true`), artifacts and documentation disagree on which calendar years are summed:

- **One interpretation (in reports/docs):** Sum **2 years**: `publication_year` through `publication_year + 1`.
- **Another (in code and eligibility):** Sum **3 years**: `publication_year` through `publication_year + 2`, with eligibility requiring data through `publication_year + 2`.

These cannot both be correct. This document traces the source of truth and attributes the inconsistency.

---

## 2. Relevant config example

```yaml
target:
  name: citations_within_2_calendar_years
  target_mode: calendar_horizon
  source: counts_by_year
  horizon_years: 2
  include_publication_year: true
  transform: log1p
```

(From `configs/experiments/baseline_representative_h2.yaml`.)

---

## 3. Source-of-truth code locations

### 3.1 Target construction

| Location | What it assumes |
|----------|-----------------|
| **`src/scholarly_outcome_prediction/features/targets.py`** | |
| `compute_calendar_horizon_target` (lines 114–139) | `horizon_years` = **index of last year** to include (last year = `publication_year + horizon_years`). When `include_publication_year` True: `start = publication_year`, `end = publication_year + horizon_years`; loop `range(start, end + 1)` → years **Y through Y+H inclusive** = **H+1 years** when including pub year. |
| Docstring (lines 122–125) | "If include_publication_year is True and horizon_years is 2: years = [Y, Y+1, Y+2]". So **3 years**. Consistent with code. |
| `build_calendar_horizon_target_column` | Calls `compute_calendar_horizon_target` and `is_horizon_eligible`; no separate year logic. |

**Observed semantics in target construction:**  
For `publication_year = Y`, `horizon_years = H`, `include_publication_year = True`:

- `start = Y`, `end = Y + H`
- Years summed: **Y, Y+1, …, Y+H** (inclusive) → **H+1** years.
- For H=2: years **2018, 2019, 2020** are summed (3 years).

### 3.2 Eligibility filtering

| Location | What it assumes |
|----------|-----------------|
| **`src/scholarly_outcome_prediction/features/targets.py`** | |
| `is_horizon_eligible` (lines 142–158) | Last year needed = `publication_year + horizon_years`. Eligible iff `publication_year + horizon_years <= max_available_citation_year`. Docstring: "we need Y, Y+1, …, Y+horizon_years; last is Y+horizon_years". |
| `prepare_df_for_target` (lines 259–262) | `eligibility_cutoff_description`: "Row eligible iff publication_year + horizon_years <= max_available_citation_year". No year-count wording. |

**Observed semantics in eligibility:**  
Require citation data through year **Y + horizon_years**. For H=2, require data through **Y+2**. This is **consistent** with the target sum (which includes year Y+2).

### 3.3 Config / settings

| Location | What it assumes |
|----------|-----------------|
| **`src/scholarly_outcome_prediction/settings.py`** (lines 62–65) | `horizon_years: int \| None = None  # e.g. 2`; `include_publication_year: bool = True  # True = citations_within_H; False = citations_in_next_H`. No definition of whether H is "number of years" or "last year offset". |

Config parsing does not impose semantics; it only passes values through.

### 3.4 Reporting

| Location | What it states |
|----------|----------------|
| **`src/scholarly_outcome_prediction/diagnostics/target_profile.py`** | |
| `build_target_semantics_description` (lines 59–62) | When `include_publication_year` True: "Target = sum of counts_by_year for years publication_year through **publication_year+{h − 1}**". For h=2: "through publication_year+1" → **2 years**. |
| `build_target_profile` → `target_semantics_note` (lines 132–135) | Same: "publication_year through publication_year + **{h − 1}** (inclusive)". For h=2: **2 years**. |
| **`README.md`** (lines 156–158) | "if `true`, the window is publication year through publication year + **(horizon_years − 1)**". For horizon_years=2: **2 years**. |

**Observed semantics in reporting:**  
Reports and README describe the window as **publication_year through publication_year + (horizon_years − 1)** = **horizon_years** years when including publication year. For horizon_years=2 that is **2 years** (Y and Y+1).

---

## 4. Worked example

**Setup:** Paper with `publication_year = 2018`. `counts_by_year` has values for 2018, 2019, 2020, 2021. `horizon_years = 2`, `include_publication_year = true`.

### 4.1 According to **implementation** (targets.py)

- `start = 2018`, `end = 2018 + 2 = 2020`.
- `range(2018, 2021)` → years **2018, 2019, 2020**.
- **Target** = sum of counts for 2018, 2019, 2020 (**3 years**).
- **Eligibility:** `last_year_needed = 2018 + 2 = 2020`. Need `2020 <= max_available_citation_year`. So we need data through **2020**. Consistent with the sum.

### 4.2 According to **reporting** (target_profile, README)

- "publication_year through publication_year + (horizon_years − 1)" = 2018 through 2019.
- **Implied target** = sum for **2018, 2019** (**2 years**).
- That would require eligibility through **2019**, not 2020. So reporting implies a **different** (and smaller) window and a **different** eligibility rule than the code.

### 4.3 If `include_publication_year = false`

- **Code:** `start = 2019`, `end = 2020`; years **2019, 2020** (2 years). Eligibility through 2020. Correct.
- **Reporting:** "next 2 full calendar years after publication year (publication_year+1 through publication_year+2)". **2 years**. Matches code. **No inconsistency** in the exclude-publication-year case.

---

## 5. Comparison: implementation vs reports

| Component | Years summed (H=2, include_publication_year=True) | Eligibility rule | Verdict |
|-----------|---------------------------------------------------|------------------|---------|
| **Target construction** (`compute_calendar_horizon_target`) | Y, Y+1, Y+2 (3 years) | N/A | Correct (and docstring matches). |
| **Eligibility** (`is_horizon_eligible`) | N/A | Through Y+2 | Correct and consistent with target. |
| **Eligibility description** (`prepare_df_for_target`) | N/A | "publication_year + horizon_years <= max" | Correct. |
| **Target profile semantics note** (`target_profile.py`) | Y through Y+1 (2 years) | Not stated in note | **Incorrect** (undercounts by one year). |
| **Target semantics description** (metrics/run text) | Y through Y+1 (2 years) | N/A | **Incorrect** (same formula). |
| **README** | Y through Y+(H−1) (2 years) | Example talks "through 2026" for 2025 paper | **Incorrect** (window wording); eligibility example is consistent with code. |

**Conclusion:** The **implementation** (target sum + eligibility) is **internally consistent**. The **bug is in reporting and documentation only**: they use "publication_year + (horizon_years − 1)" for the last year when include_publication_year is true, but the code uses "publication_year + horizon_years" as the last year (so H+1 years are summed).

---

## 6. Test coverage

| Test | File | What it encodes | Pass? |
|------|------|-----------------|-------|
| `test_compute_calendar_horizon_include_publication_year` | `tests/test_calendar_horizon_targets.py` | For Y=2018, H=2, inc=True: sum = 1+2+3 (years 2018, 2019, 2020). | Yes. Encodes **3 years**. |
| `test_compute_calendar_horizon_exclude_publication_year` | Same | For H=2, inc=False: sum Y+1, Y+2. | Yes. |
| `test_is_horizon_eligible` | Same | Eligibility via `last_year_needed = pub_year + horizon_years`; e.g. 2019, max 2020, H=2, inc=True → False (need 2021). | Yes. Matches implementation. |

**Findings:** Tests **agree with the implementation** (3 years summed when H=2 and include_publication_year True; eligibility through Y+2). They do **not** encode the reporting formula (Y through Y+1). So tests **do not** protect the incorrect report wording; they protect the **current code behavior**. No off-by-one in tests; the wrong text is only in reporting/docs.

---

## 7. Final diagnosis

### Case A: Implementation correct, reporting wrong — **YES**

- **Target computation:** Correct. Sums Y through Y+horizon_years (inclusive) when include_publication_year True; i.e. **horizon_years+1** years.
- **Eligibility:** Correct and aligned with target (requires data through Y+horizon_years).
- **Reporting (target_profile semantics note, build_target_semantics_description, README):** **Wrong.** They say "publication_year through publication_year + (horizon_years − 1)", which describes **horizon_years** years, not **horizon_years+1**. So for horizon_years=2 they describe 2 years while the code sums 3 years.

### No bug in

- Target construction logic.
- Eligibility logic.
- Eligibility cutoff description string (it correctly says "publication_year + horizon_years").
- Tests (they match implementation).

### Bug in (wording only)

- `diagnostics/target_profile.py`: `target_semantics_note` and `build_target_semantics_description` when `include_publication_year` is True. They use `(h - 1)` for the last year; should describe last year as `publication_year + horizon_years` (and optionally state that this is H+1 years when including publication year).
- `README.md`: Same: "publication year + (horizon_years − 1)" should be "publication year + horizon_years" for the *last* year when include_publication_year is true (and clarify that the number of years is horizon_years+1 in that case).

---

## 8. Recommended fix strategy (high-level)

1. **Do not change** target computation or eligibility logic; they are consistent and tests encode them.
2. **Fix reporting and docs** so they match the implementation:
   - **Target profile / semantics description:** For `include_publication_year true`, state that the target is the sum of counts for years **publication_year through publication_year + horizon_years** (inclusive), i.e. **horizon_years+1** calendar years. Avoid the formula "publication_year + (horizon_years − 1)" for the last year.
   - **README:** Same: describe the window as publication year **through** publication year **+ horizon_years** (inclusive) when include_publication_year is true, and note that this is horizon_years+1 years (e.g. "2" in the config means last year is Y+2, so 3 years summed).
3. Optionally add a single test or assertion that the **reported** semantics string (if parsed or checked in tests) matches the implementation (e.g. that for H=2 the summed years are 3 when include_publication_year is True).

---

## 9. Minimal reproducible evidence (snippet)

```python
# Current implementation behavior (features/targets.py)
# publication_year=2018, horizon_years=2, include_publication_year=True
start = 2018
end = 2018 + 2  # 2020
years_summed = list(range(2018, 2020 + 1))  # [2018, 2019, 2020] → 3 years
# Eligibility: last_year_needed = 2018 + 2 = 2020 (must have data through 2020)
```

Reporting currently says: "publication_year through publication_year + (2-1)" = 2018 through 2019 = **2 years**. So report and code disagree by one year.

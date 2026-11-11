# Real Estate data and feature dictionary

The committed training and test CSVs contain geographic aggregate observations. The capstone concerns U.S. housing, mortgage, and demographic measures. The original external publisher, survey vintage, and authoritative field glossary are unverified. Descriptions below follow the capstone and code. Confirm definitions and timing against an authoritative source before external use.

| Field or group | Role and interpretation |
|---|---|
| `UID` | Observation identifier used for indexing and split overlap checks. Excluded from predictors. |
| `hc_mortgage_mean` | Target. Mean monthly mortgage and owner cost for the observation, in the source's monetary units. Rows without the target are excluded from supervised training and evaluation. |
| `pop` | Population count for the geographic observation. |
| `family_mean` | Mean family income in the source monetary units. |
| `second_mortgage`, `home_equity`, `debt` | Source housing and debt indicators. Exact denominators and observation dates require source confirmation. |
| `hs_degree`, `pct_own` | Source education and home ownership measures. Confirm denominators and encoding before reuse. |
| `married`, `separated`, `divorced` | Source marital status measures. Confirm denominators and encoding before reuse. |
| `male_age_median`, `female_age_median` | Source median age measures. Their arithmetic mean produces the numeric `age_median` feature. This derived feature is not a population-weighted median. |
| `STATEID`, `type` | State identifier and place type, treated as categorical predictors. |
| `COUNTYID`, `zip_code`, `BLOCKID`, `SUMLEVEL` | Geographic or source identifiers and codes. They are outside the modern predictor set. |

The model assumes geographic summaries are available when scoring. The repository does not establish that the income, debt, or ownership fields precede the target in time. Verify availability and any shared derivations before using the model for forecasting or decisions about lending. See [feature selection and preprocessing](../src/modeling.py).

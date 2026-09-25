# Real Estate & Mortgage Analytics Capstone

## Project Overview

This archived capstone analyzes U.S. geographic, housing, mortgage, income, and demographic data in a banking and mortgage analytics scenario. It combines data preparation, exploratory data analysis, geographic analysis, feature engineering, factor analysis, regression modeling, and Tableau reporting. It is a learning project, not a production banking system.

## Business Problem

The project explores mortgage-backed and real-estate trends across geographic areas. Its analytical tasks compare family income, rent, home ownership, mortgage costs, and household debt-related indicators; model monthly mortgage and owner costs; and communicate the analysis through a dashboard.

## Dataset

The tracked training and test CSV files contain geographic observations and variables covering:

- geographic identifiers and population;
- household and family income;
- rent, mortgage, and homeowner costs;
- second mortgages, home equity, debt, and home ownership;
- education and age demographics; and
- marital-status indicators.

The notebook treats `UID` as the observation identifier and sets it as the index during preparation. The original external data source is not explicitly documented in this archived project repository.

## Analysis Workflow

### Data preparation

The notebook:

- loads the training and test datasets;
- inspects shapes, columns, and descriptive statistics;
- identifies `UID` as the index;
- examines missingness and uses mean imputation for numerical missing values;
- inspects feature variance; and
- removes `BLOCKID` and `SUMLEVEL` from the working datasets.

### Debt and housing analysis

The analysis identifies locations with relatively high second-mortgage prevalence, maps geographic results with Plotly, and derives a `bad_debt` feature. It also creates debt-related pie charts, compares debt indicators for selected cities, and examines household and family income distributions.

### Demographic analysis

The notebook explores population distributions, engineers population-density and median-age fields, creates population bins, and analyzes marital-status indicators. It also compares rent with family income and examines correlations among selected variables.

### Factor analysis

Factor analysis is applied to numerical features to explore possible latent relationships. The archived workflow does not establish that the resulting factors were fully validated or statistically interpreted.

### Regression modeling

A Scikit-learn Linear Regression model predicts `hc_mortgage_mean`, the mean monthly mortgage and owner costs for a geographic observation. Predictor groups include geographic identifiers, population, family income, mortgage and debt measures, education, age, ownership, and marital-status indicators.

## Tableau Dashboard

The repository includes a packaged Tableau workbook. Dashboard-related tasks represented in the project include:

- average rent by place type;
- debt and bad-debt views;
- geographic second-mortgage analysis;
- correlation visualization; and
- population distribution across place types.

The notebook records a [historical Tableau Public dashboard link](https://public.tableau.com/profile/saurabh.zambare#!/vizhome/RealEstate_16057645309010/RealEstate?publish=yes). Its current availability has not been verified as part of this repository cleanup. The packaged workbook is available at [`tableau/real_estate_dashboard.twbx`](tableau/real_estate_dashboard.twbx).

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Plotly
- pandasql
- Scikit-learn
- factor-analyzer
- Jupyter Notebook
- Tableau

## Repository Structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── real_estate_capstone.ipynb
├── tableau/
│   └── real_estate_dashboard.twbx
└── docs/
    └── problem_statement.docx
```

- `data/` contains the intentionally tracked training and test datasets.
- `notebooks/` contains the original analytical workflow, with only dataset paths adjusted for the new layout.
- `tableau/` contains the packaged Tableau dashboard workbook.
- `docs/` contains the original project problem statement.
- `requirements.txt` lists the unpinned Python packages needed by the notebook.

## Running the Notebook

1. Clone and enter the repository:

   ```bash
   git clone https://github.com/Saurabhzambare/Real_Estate_Capstone.git
   cd Real_Estate_Capstone
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, use `.venv\Scripts\Activate.ps1` instead.

3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `notebooks/real_estate_capstone.ipynb` and run it from the `notebooks/` directory so its relative data paths resolve correctly.

This is an archived learning project created with older library versions. Some cells may require compatibility updates when run with modern packages.

## Historical Modeling Result

Stored notebook output for the overall linear-regression experiment reports approximately:

- **R²:** 0.7348
- **RMSE:** 323.10

These values are historical notebook outputs, not validated production-quality performance metrics. Limitations in the original methodology would need to be corrected before treating them as a rigorous benchmark.

## Current Limitations

- This is an older learning and capstone project.
- The test feature matrix is scaled with a separate `fit_transform()` rather than transformed with the scaler fitted on the training data.
- The section described as state-level modeling filters rows using `COUNTYID` rather than `STATEID`.
- Model validation is limited, and there is no cross-validation pipeline.
- The workflow is notebook-oriented rather than productionized.
- Legacy APIs such as `seaborn.distplot()` are used.
- The original package versions were not recorded.
- Some dependency installation commands are embedded directly in notebook cells.
- The original external dataset source is not clearly documented in the repository.

These historical implementation details are documented rather than silently changed during this presentation-focused cleanup.

## Future Improvements

The following are modernization opportunities, not functionality currently provided by the repository:

- fit preprocessing on training data and use that fitted transformer for test data;
- correct the state-level filtering logic;
- create a Scikit-learn Pipeline;
- introduce a train/validation strategy or cross-validation;
- review feature selection and multicollinearity;
- compare additional regression models;
- replace deprecated plotting APIs;
- improve factor-analysis validation and interpretation;
- pin package versions for reproducibility;
- extract reusable preprocessing and modeling code into `src/`;
- add automated tests;
- export high-quality Tableau dashboard screenshots for GitHub; and
- verify or update the Tableau Public sharing URL.

## Key Skills Demonstrated

- Exploratory data analysis and data cleaning
- Feature engineering
- Geographic visualization
- Statistical modeling and linear regression
- Model evaluation
- Dimensional and factor analysis
- Business-oriented data interpretation
- Tableau dashboard development

## Author / Portfolio Context

This repository is maintained as part of Saurabh Zambare's data science and software-development portfolio. It preserves an earlier capstone project while improving its organization, presentation, and documentation for portfolio review.

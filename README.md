# Wildfire Susceptibility and Hazard Modelling

> Computational workflow developed for the Master's dissertation<br>
> **_Modelação da perigosidade de incêndio rural: comparação e validação de diferentes abordagens_**

**Ana Luísa Pereira da Costa**<br>
Master's Degree in Geospatial Information Engineering<br>
University of Coimbra

---

## Overview

This repository contains the computational workflow developed to model, validate and analyse wildfire susceptibility and hazard in the context of a Master's dissertation in Geospatial Information Engineering.

The work combines statistical and machine-learning approaches to assess structural and seasonal components of wildfire hazard.

The repository includes workflows for:

- preparation and harmonisation of geospatial datasets;
- wildfire susceptibility modelling using **Likelihood Ratio (LR)**;
- hybrid susceptibility modelling using **Likelihood Ratio values and Random Forest (LRi–RF)**;
- historical wildfire probability estimation;
- structural wildfire hazard assessment;
- seasonal modelling based on the **Seasonal Severity Rating (SSR)**;
- temporal and spatial validation;
- comparison between modelling scenarios;
- analysis of the spatial distribution of the 2025 burned areas.

The methodology was developed using regional calibration areas in Portugal and Spain and subsequently analysed at local study-area scale.

---

## Contents

- [Repository structure](#repository-structure)
- [Workflow](#workflow)
- [Data preparation](#data-preparation)
- [Likelihood Ratio](#likelihood-ratio)
- [LRi–Random Forest](#lrirandom-forest)
- [Seasonal modelling](#seasonal-modelling)
- [Analysis](#analysis)
- [Python utilities](#python-utilities)
- [Computational environment](#computational-environment)
- [Running the environment](#running-the-environment)
- [Data](#data)
- [Reproducibility notes](#reproducibility-notes)
- [Dissertation version](#dissertation-version)
- [Project context](#project-context)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Repository structure

```text
inescc-firerisk/
├── notebooks/
│   ├── prep_data/
│   ├── lr/
│   ├── rf/
│   ├── ssr/
│   └── analysis/
│
├── scripts/
│
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── LICENSE
└── README.md
```

| Directory | Purpose |
|---|---|
| `notebooks/prep_data/` | Preparation and harmonisation of input datasets |
| `notebooks/lr/` | Likelihood Ratio susceptibility and hazard workflow |
| `notebooks/rf/` | Hybrid LRi–Random Forest workflow |
| `notebooks/ssr/` | Seasonal modelling and 2025 validation |
| `notebooks/analysis/` | Validation, comparison and territorial analyses |
| `scripts/` | Reusable Python functions used throughout the workflows |

---

## Workflow

The repository follows the main stages of the methodology developed in the dissertation.

```text
Input geospatial data
        │
        ▼
Data preparation and harmonisation
        │
        ├───────────────┐
        │               │
        ▼               ▼
Likelihood Ratio     LRi–Random Forest
        │               │
        ▼               ▼
Structural          Machine-learning
susceptibility      susceptibility
        │               │
        └───────┬───────┘
                │
                ▼
     Historical wildfire probability
                │
                ▼
      Structural hazard assessment

Meteorological information
        │
        ▼
Seasonal Severity Rating (SSR)
        │
        ▼
Seasonal modelling
        │
        ▼
Validation and territorial analysis
        │
        ▼
Scenario comparison and 2025 assessment
```

The analysis notebooks combine the outputs produced by the different modelling components and support the comparisons presented in the dissertation.

---

## Data preparation

Directory:

```text
notebooks/prep_data/
```

The data-preparation workflow contains:

```text
01_prep_aoi.ipynb
02_prep_aa_icnf.ipynb
02b_prep_aa_effis.ipynb
03a_harm_table.ipynb
03b_harm_lulc.ipynb
03c_harm_siose_ar.ipynb
04_prep_lulc_rst.ipynb
```

These notebooks support:

- preparation of the study areas;
- processing of burned-area datasets;
- preparation of ICNF and EFFIS burned-area information;
- thematic harmonisation of land-use and land-cover datasets;
- temporal harmonisation of land-use and land-cover information;
- preparation of raster inputs used by the modelling workflows.

---

## Likelihood Ratio

Directory:

```text
notebooks/lr/
```

The Likelihood Ratio workflow contains:

```text
01_var_reclass.ipynb
02_calc_perigosity.ipynb
03_valida.ipynb
```

This workflow supports:

- reclassification of explanatory variables;
- calculation of Likelihood Ratio values;
- wildfire susceptibility modelling;
- integration with historical wildfire probability;
- structural hazard calculation;
- model validation.

The structural explanatory variables considered in the dissertation include elevation, slope and harmonised land use and land cover.

---

## LRi–Random Forest

Directory:

```text
notebooks/rf/
```

The hybrid LRi–RF workflow contains:

```text
01_create_rf_target.ipynb
02_split_train_test.ipynb
03_excel_models.ipynb
04_train.ipynb
05_class.ipynb
06_calc_peri.ipynb
07_valida.ipynb
```

The approach uses the Likelihood Ratio values associated with the classes of the explanatory variables as predictors in Random Forest models.

The notebooks cover:

- target preparation;
- presence and pseudo-absence sampling;
- train/test preparation;
- model configuration;
- Random Forest training;
- spatial classification;
- hazard calculation;
- model validation.

Multiple model replicas are used to assess the stability and variability of the Random Forest results.

---

## Seasonal modelling

Directory:

```text
notebooks/ssr/
```

The seasonal workflow contains:

```text
01_prep_ssr_abs.ipynb
02_prep_burned_targets_yearly.ipynb
03_build_common_samples.ipynb
04_train_logistic_models.ipynb
05_apply_models_2025.ipynb
06_validate_models_2025.ipynb
```

The seasonal component evaluates the relationship between meteorological conditions before the summer season and subsequent wildfire occurrence.

The workflow includes:

- preparation of Seasonal Severity Rating information;
- preparation of annual burned-area targets;
- construction of common model samples;
- logistic-regression model training;
- application of the models to 2025;
- independent validation using the 2025 fire season.

---

## Analysis

Directory:

```text
notebooks/analysis/
```

The analysis workflow contains:

```text
00_regional_validation.ipynb
01_model_interpretation.ipynb
02_regional_classification_and_pilot_clips.ipynb
03_pilot_class_distribution.ipynb
04_spatial_comparisons.ipynb
05_priorities_agreement.ipynb
06_rf_replica_uncertainty.ipynb
07_ssr_territorial_effect.ipynb
08_burned_area_2025_by_class.ipynb
```

These notebooks support:

- regional model validation;
- interpretation of model behaviour;
- regional classification of model outputs;
- extraction and analysis of local study areas;
- analysis of class distributions;
- spatial comparison between modelling scenarios;
- assessment of agreement between priority areas;
- analysis of Random Forest replica variability;
- assessment of the territorial effect of the seasonal component;
- analysis of the distribution of the 2025 burned areas across hazard classes.

---

## Python utilities

Reusable processing functions are stored in:

```text
scripts/
```

The modules support operations including:

- raster processing;
- study-area preparation;
- burned-area processing;
- land-use and land-cover harmonisation;
- sampling;
- Random Forest target preparation;
- susceptibility and hazard calculations;
- classification;
- validation;
- spatial comparison;
- model interpretation;
- territorial analysis;
- seasonal analysis.

Separating reusable functions from the notebooks reduces code duplication and keeps the main processing workflows easier to inspect and maintain.

---

## Computational environment

The computational environment is defined through Docker.

The main components include:

- **Ubuntu 24.04**
- **Python 3**
- **GRASS GIS 8.4**
- **GDAL**
- **JupyterLab**
- scientific and geospatial Python libraries
- **GLASS**

The Docker image installs **GRASS GIS 8.4.2** from source.

The [GLASS](https://github.com/jasp382/glass) geospatial library is also installed during the Docker build and is used by several processing steps in the workflow.

---

## Running the environment

### 1. Clone the repository

```bash
git clone https://github.com/analuisa1702/inescc-firerisk.git
cd inescc-firerisk
```

### 2. Build and start the Docker environment

```bash
docker compose up --build
```

JupyterLab is exposed at:

```text
http://localhost:8889
```

The repository is mounted inside the container at:

```text
/code
```

> [!IMPORTANT]
> The current `docker-compose.yml` contains an external-storage bind mount:
>
> ```text
> /Volumes/NO NAME/ → /code/pen
> ```
>
> This path corresponds to the local development environment used during the dissertation.
> It must be adapted or removed when running the repository on another machine.

---

## Data

The datasets used in the dissertation are **not stored in this repository**.

This is mainly due to:

- dataset size;
- external data-source requirements;
- source-specific access and distribution conditions.

The local directory:

```text
data/
```

is explicitly excluded from version control through `.gitignore`.

The modelling workflow uses geospatial information including:

- digital elevation data;
- derived topographic variables;
- land-use and land-cover datasets;
- historical burned-area records;
- meteorological information used in the seasonal component.

The corresponding preparation and harmonisation procedures are implemented primarily under:

```text
notebooks/prep_data/
```

Reproducing the complete workflow therefore requires obtaining the corresponding source datasets and reproducing the expected local directory structure.

---

## Reproducibility notes

This repository preserves the computational implementation used to support the methodology and results presented in the dissertation.

The source datasets are not distributed with the repository, and full execution therefore depends on access to the corresponding input data.

Some notebooks contain paths associated with the computational environment and storage structure used during development. These paths may need to be adapted when the workflow is executed on another system.

> [!NOTE]
> The repository should be interpreted as the computational implementation associated with the dissertation rather than as a self-contained distribution of all input datasets.

The Docker environment is provided to preserve the main software dependencies used during development and to improve computational reproducibility.

---

## Dissertation version

The repository state associated with the submitted Master's dissertation is preserved under the Git tag:

```text
v1.0-dissertation
```

This tag provides a fixed reference to the documented code version associated with the dissertation.

Future development may continue on the `main` branch without modifying this archived version.

---

## Project context

This work was developed within the **Master's Degree in Geospatial Information Engineering** at the **University of Coimbra**.

The dissertation is related to the **SenForFire** project, which addresses wildfire monitoring, prevention and early-warning approaches in the SUDOE territory.

---

## Acknowledgements

Part of the initial codebase and methodological support used as a starting point for this work was provided by my supervisor, **Joaquim A. S. Patriarca**.

His contributions provided an important technical basis for the subsequent development and adaptation of the workflows implemented in this repository.

- GitHub: [jasp382](https://github.com/jasp382)
- GLASS geospatial library: [jasp382/glass](https://github.com/jasp382/glass)

The final workflows, adaptations, scenario implementation, validation procedures and analyses contained in this repository were developed in the context of the Master's dissertation.

---

## License

Licensing information for this repository is provided in the [`LICENSE`](LICENSE) file.

Third-party libraries and dependencies used by the project remain subject to their respective licenses.

The presence of a third-party dependency in the computational environment does not replace or modify the licensing terms applicable to that dependency.

---

## Author

**Ana Luísa Pereira da Costa**<br>
Master's Degree in Geospatial Information Engineering<br>
University of Coimbra

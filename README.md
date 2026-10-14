# GeoCTGAN

## GeoCTGAN generates privacy-aware geosynthetic event data

## Installation

Python 3.10 or 3.11.

```bash
python -m venv .venv
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
python -m pip install -r requirements.txt
python -m jupyter lab
```

## Data inputs

| Analysis | Input |
|---|---|
| Toronto | Major Crime Indicators point-event data |
| Toronto | Road-network GeoPackage with source `u` and `v` endpoint identifiers |
| Toronto | Official 2021 Dissemination Area polygons |
| Toronto | 2021 Ontario Marginalization Index and DA population table |
| Ottawa | Offences point-event data, 2018–2024 |
| Ottawa | City of Ottawa Road Centrelines |
| Ottawa | Official 2021 Dissemination Area polygons and population table |
| Ottawa | Neighbourhood boundary extent used for the common spatial grid |

Data are supplied separately. See the main manuscript's **Data availability** and **Methods** sections, and **Section 10 of the Supplementary Information**, for data sources and details of the Ottawa analysis.

For terminal runs, specify input paths, archive-member paths and layer names in the configuration files. For notebook runs, provide paths at the prompts; update the Toronto archive-member and layer settings where necessary.

### Ottawa input preparation

Original study filenames and directory structures are not required. At the Ottawa notebook prompts, provide the crime, road and study-extent shapefile paths, DA boundary archive and population table. Shapefile paths may be absolute or relative to the supplied input directory or ZIP root. Keep the components of each shapefile together, including its coordinate-system information.

The notebook reads the required crime and road attributes from the source files and derives the modelling variables during preparation. Retain the source variables expected by the notebook; if an export uses different field names, align those names before running it. The DA population input is a prepared table derived from Census data and must follow the column structure specified in the notebook.

Official DAs are used for area-level event counts and spatial-autocorrelation analyses. Any neighbourhood boundaries referred to in the code supply only the bounding rectangle for the common KDE and square-grid evaluations; they are not used as the event-aggregation units. Retain this spatial extent when reproducing the reported grid-based results.

## Execution

| Notebook | Analysis |
|---|---|
| `GeoCTGAN_Run.ipynb` | Primary city (Toronto) configuration selection, generation, benchmarks and evaluation |
| `GeoCTGAN - Benchmarking and Ablation Analyses.ipynb` | Benchmark generation   |
| `Geosynthetic Data Evaluation.ipynb` | Statistical, temporal, spatial, downstream and privacy evaluation |
| `GeoCTGAN_Ottawa_Generation_and_Evaluation.ipynb` | Second city (Ottawa) transfer analysis |
| `notebooks/00_Configuration_Selection.ipynb` | Toronto configuration selection |
| `notebooks/01_GeoCTGAN_Generation.ipynb` | Toronto generation using a selected configuration |
| `notebooks/02_Geosynthetic_Data_Evaluation.ipynb` | Toronto held-out evaluation |

The notebooks in `notebooks/` provide separate selection, generation and evaluation entry points as an alternative to `GeoCTGAN_Run.ipynb`.

Users should use a fresh output directory whenever inputs or settings change.

## Terminal execution
Run these commands from the repository’s root directory. Replace the placeholder paths in the configuration files and commands with your own paths. After configuration selection finishes, set SELECTED_CONFIG_PATH in both configuration templates to the generated Metadata/selected_config.json file.
```bash
python -m scripts.run_sensitivity configs/sensitivity.json
python -m scripts.run_pipeline configs/definitive.template.json
python -m scripts.run_benchmarks configs/benchmark.template.json TRAINING_TABLE OUTPUT_DIRECTORY
python -m scripts.run_evaluation GENERATION_DIRECTORY --output-parent OUTPUT_DIRECTORY --da-boundaries DA_FILE --marginalization-csv MARGINALIZATION_FILE --benchmark-manifest BENCHMARK_MANIFEST_CSV
```

## Computational tests

```bash
python -m pytest tests
```

## License

[MIT](LICENSE)

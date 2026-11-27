# DOELG task guide

DOELG constructs disaster–observation event logic from text, instantiates that knowledge for a regional disaster scenario, maps spatial observation requirements, and evaluates the results. Task numbers below follow `run.py`: **3.4** means main menu 3, subtask 4.

- [Start the program](#start-the-program)
- [Tasks and paper correspondence](#tasks-and-paper-correspondence)
- [Execution order and dependencies](#execution-order-and-dependencies)
- [Parameters and commands](#parameters-and-commands)
- [Configuration and outputs](#configuration-and-outputs)

## Start the program

On Windows x64, extract `DOELG_windows.zip` and double-click `Start_DOELG.bat`. The first launch downloads Python 3.13 and installs the locked dependencies; subsequent launches reuse `.runtime`. Keep the project in a writable folder. Internet access is required for initial setup. Setup errors are recorded in `.runtime/setup.log`.

Alternatively, from the project root with an existing environment:

```powershell
conda activate KGC
python run.py
```

Before the menu, enter the Neo4j server URI, username, password and DeepSeek API Key. Press Enter to retain an existing value; password/Key input is hidden. Service settings are saved to `construct_config.json` and passed to all tasks in that menu session. Task-specific parameters apply only to that invocation. The launcher does not start Neo4j, Ollama, Nominatim or an LLM service.

The shared configuration stores passwords and API keys encrypted. Keep `.doelg-credentials.key` beside it. Automatic defaults require this bundled key, so the protection hides JSON plaintext but cannot prevent a package recipient from extracting credentials.

Use `Start_DOELG.bat --check` to prepare/check the environment without opening the menu. `python run.py --dry-run` previews task commands and does not save service settings or run tasks. Enter `0` to go back/exit; Ctrl+C cancels input or a running task. Blank optional task parameters use the selected script/configuration defaults. Required input paths must exist; preparation requires a new output workbook. Direct `main/` commands below do not show the startup configuration prompts.

## Tasks and paper correspondence

References follow the supplied anonymous manuscript, *Event Logic Graph for Spatial Observation Requirement Causal-Geospatial Inference in Multi-Hazard Flood Scenario*.

### 1. Extraction and fusion

Entry point: `main/main_extraction.py`. These stages construct **DOELG-history** in the selected Neo4j database (Section 4.2.1; Figure 13).

| Task | Function | Method section |
| --- | --- | --- |
| **1.1 `extract`** | Extract geographic objects → disaster/observation events → relations; apply Appendix E causal filtering before insertion | 3.3.1 |
| **1.2 `fuse`** | Merge coreferential arguments/events and deduplicate relations | 3.3.1 |
| **1.3 `env`** | Add structured hazard-formative environment descriptions | 3.3.1; input to 3.4.1 |
| **1.4 `obs_pat`** | Add observation patterns using corpus context and sensor references | 3.3.1; input to 3.4.2 |
| **1.5 `spatial`** | Geocode objects and calculate geometric, basin and semantic/type-path relationships | 3.3.2 |
| **1.6 `all`** | Run 1.1 → 1.2 → 1.3 → 1.4 → 1.5 | 3.3 |

### 2. Knowledge generalization

Entry point: `main/main_generalization.py`. These stages add **DOELG-abstract** knowledge to the selected history database (Section 4.2.2; Figure 14). They do not copy it automatically to a separate database named `DOELG-abstract`.

| Task | Function | Method section |
| --- | --- | --- |
| **2.1 `disaster_gen`** | Cluster historical disaster pairs; create MergeDisasterEvent/GlobalDisasterEvent hierarchy and causal confidence | 3.4.1 |
| **2.2 `spatial_rule`** | Refine spatial propagation and quantitative environment rules | 3.4.1, 3.4.3 |
| **2.3 `obs_gen`** | Align standard factors; construct generalized observations and direct/inherited triggers | 3.4.2 |
| **2.4 `stats`** | Export pattern counts, causal chains, hub statistics and disaster–observation summaries | 4.2.2 graph statistics |
| **2.5 `all`** | Run 2.1 → 2.2 → 2.3 → 2.4 | 3.4 |

### 3. Instantiation and mapping

Entry point: `main/main_instantiation.py`. Reads abstract knowledge and writes a separate **DOELG-instance** database.

| Task | Function | Paper correspondence |
| --- | --- | --- |
| **3.1 `instantiate`** | Infer regional disaster paths and triggered observations from initial polygon events | 3.5.1; 4.3.1, Figures 15–16 and graph data for Table 5 |
| **3.2 `map`** | Generate hazard/factor GeoTIFFs, optional PNGs and cell summaries from an existing graph/snapshot | 3.5.2; 4.3.1, Figures 17–19 |
| **3.3 `all`** | Run 3.1 followed by 3.2 | 4.3.1 |
| **3.4 `update` — Update and map** | Copy a baseline instance, infer new paths from additional initial events, and map the combined graph | 4.3.2 incremental experiment, Figure 22 |

Update preserves the baseline and requires a different, empty destination database. Its default input is the nine-event `data/geo_dataset/events_zhengzhou/update.geojson`. Initial-event features must have an English `properties.event` matching the abstract graph and valid Polygon/MultiPolygon geometry in EPSG:4326/CRS84. If an abstract pattern lacks an environment rule, instantiation may call the configured model and save the generated rule to the logic graph.

### 4. Extraction validation

Entry point: `main/main_extraction_validation.py`. Uses local workbooks to evaluate the extraction method in Section 3.3.1 and reproduce **Table 4** in Section 4.2.1; no live model/database requests.

| Task | Function | Required input |
| --- | --- | --- |
| **4.1 `evaluate`** | Report extraction metrics for one target/workbook | Prepared workbook |
| **4.2 `plot`** | Compare default/code methods across few-shot settings | One target's few-shot workbook directory |
| **4.3 `table`** | Export all three Table 4 F1 metrics for both methods | Dataset parent containing `event`, `geo`, `event_relation` |
| **4.4 `prepare`** | Parse existing extraction strings into JSON columns in a new workbook | Unprepared workbook; does not perform new LLM extraction |

Targets are `event`, `geo` and `relation`. Evaluation expects `raw_json` and `true_json`. Event-F1 combines subject/object and argument matches; Geo/Spatial F1 combines entity/relation matches. Preparation can fall back to predictions when ground truth is missing, so verify annotations before interpreting scores. Bundled workbooks are already prepared.

### 5. Method validation and paper reproduction

Entry point: `main/main_validation.py`. OELG supplies the implemented comparison baseline described in Section 4.3.2.

| Task | Function and source | Paper correspondence |
| --- | --- | --- |
| **5.1 `instantiate`** | Construct OELG instances from causal knowledge and observation knowledge; generate maps by default | Baseline for 4.3.2 |
| **5.2 `map`** | Map an existing OELG result in its instance database | Baseline fields for Figures 20–21 |
| **5.3 `compare`** | Compare DOELG/OELG fields against validation points; reuse complete rasters and generate missing sets from existing graphs | 4.3.2, Figures 20–21; Tables 6–7 |
| **5.4 `pattern`** | Choose generalized Neo4j graph statistics or frozen archived pattern counts | 4.2.2, including the archived 1,656-pattern result |
| **5.5 `reproduction`** | Reconstruct fields and verify numerical results from frozen inputs only | Table 4, pattern counts, Tables 6–7 and comparison curves |

Comparison reports AUCPA, HR@10%, concentration relative to random selection and Area@80%Hit. The subsidence point set/layer is `Subsidence`; it is distinct from the `Structural Collapse` hazard.

Task 5.3 does not ask for an OELG snapshot name: when mapping is needed it discovers the latest complete supported result. It never instantiates an empty graph or substitutes archived paper results. Complete raster sets can be compared without Neo4j; known source-database/snapshot mismatches prevent reuse, and missing source metadata is reported as unverified. Reused rasters reflect their export time. After graph changes, select fresh raster roots and remove the relevant generated cache under `data/database_validation/maps` if needed.

## Execution order and dependencies

### Choose a workflow

| Goal | Order | Requirements |
| --- | --- | --- |
| Verify archived paper values | **5.5** directly; **4.3** for Table 4 only; **5.4 → Existing archives** for patterns only | Bundled inputs; no Neo4j, embedding, geocoding or LLM services after environment setup |
| Infer/map using supplied abstract graphs | Import dumps → **3.3** (or **3.1 → 3.2**) | `DOELG-abstract`, local scenario inputs and a separate instance destination |
| Construct new knowledge from text | **1.6 → 2.5 → 3.3** | Use the same history database for menus 1–2; select it explicitly as menu 3's source logic database |
| Compare live methods | **3.3** and **5.1** → **5.3** | Same scenario/depth/grid; populated instance graphs or complete compatible rasters for both methods |
| Map OELG separately | **5.1 → 5.2** | Needed only if maps were skipped or must be redrawn; 5.3 can also generate missing maps |
| Incrementally update a scenario | Baseline **3.1/3.3 → 3.4** | Existing baseline, compatible abstract logic, new events, separate empty update database |
| Evaluate new extraction workbooks | **4.4 → 4.1/4.2/4.3** | Verified ground truth; skip 4.4 for already prepared workbooks |
| Inspect live abstract patterns | **2.1** (normally **2.5**) → **5.4 → Local Neo4j databases** | Select the generalized logic database, not an instance database |

Each `all` covers only its own menu group. Selecting a database in one group does not select it automatically in another. For separate stage runs, 1.2–1.5 require extracted records; 2.2 requires the hierarchy and cluster file from 2.1; 2.3 requires abstract disasters plus extracted observations/factors. Complete observation-pattern and spatial stages before full generalization. Run 2.4 after 2.3 for observation coverage statistics. Keep cluster files and graph versions matched; use separate configured export paths for independent datasets.

The default extraction input is the small `sample_corpus`, intended for demonstration. It does not reproduce the full 13,727-sentence paper experiment or guarantee its graph counts. The larger bundled corpus is `wiki_corpus`; explicitly select and verify the intended input for a full run. Live results depend on corpus, graph/model versions and settings. Task 5.5 verifies frozen paper values: 24 Table 4 values, 80 field metrics and the archived pattern summary. It does not rebuild historical graphs or perform Figure 22's update. Tasks generate graph content/maps/reports; manuscript screenshot layouts and formatted tables may need separate composition.

### Databases and services

Use Neo4j 5.24.0 or later with APOC. For supplied knowledge, import `Geo-type.dump`, `DOELG-abstract.dump` and `OELG-abstract.dump` into the same Neo4j instance under the corresponding names. `geocoding.type_database` selects the type graph (supplied `Geo-type`). Menus 1–2 can be skipped when the required abstract knowledge is imported.

| Work | Additional requirements beyond Python |
| --- | --- |
| Menu 1 | Neo4j; configured extraction/enrichment LLM APIs; Ollama embeddings for fusion/sensor retrieval; sensor workbook/index for `obs_pat`; Nominatim endpoints, basin data and type graph for `spatial` |
| Menu 2 | Existing history graph; embeddings and LLMs for generalization stages; `stats` reads existing graph content |
| Menu 3 instantiate/update | Neo4j, initial-event GeoJSON, geographic feature layers and enabled basin layer; model access for missing environment rules |
| Menu 3 map / 5.2 | Existing graph and local mapping inputs; DOELG `--snapshot` permits offline mapping |
| 5.1 | OELG causal graph, observation knowledge and local scenario layers; supplied strict profile uses stored environment rules |
| Menu 4 / archive 5.4 / 5.5 | Local workbooks/snapshots/geometries/reference files only |

Missing write-target databases are created in the running Neo4j instance for menu 1, DOELG instantiation/update and OELG instantiation. This requires multiple-database support and creation privileges. Existing databases are reused without clearing; source databases and map-only graphs must already exist. Defaults use Shanghai date `YYYYMMDD`: `DOELG-history-demo-{date}`, `DOELG-instance-{date}`, `OELG-instance-{date}`; the update default adds `-update`. Select an explicit earlier name when continuing another day's run.

## Parameters and commands

Run commands from the project root. Menu prompts expose common options; additional options are available directly. `python main/<script>.py --help` lists the accepted arguments. CLI overrides take precedence over the selected task configuration. Startup credentials apply throughout the interactive session; direct commands read configuration/environment values.

### Extraction and generalization

| Argument | Meaning / default | Applies to |
| --- | --- | --- |
| `-t, --task` | Required; use the task names in menus 1 or 2 above | Both |
| `--config` | Construction/model/connection settings; `construct_config.json` | Both |
| `-db, --database` | History/generalization database; `extraction.database` or `generalization.database` | Both |
| `-w, --workers` | Worker count; corresponding section's `workers` (supplied `20`) | Both; individual helper limits can also be configured |
| `-i, --input_dir` | Input JSON directory; `extraction.input_dir` | Menu 1 `extract` / `all` |

```powershell
python main/main_extraction.py -t all -db my-history -i data/extraction_source_dataset/sample_corpus
python main/main_generalization.py -t all -db my-history
```

Both menus prompt for configuration → database → workers; extraction also prompts for input directory on `extract`/`all`. Their configured data paths resolve from the project root.

### DOELG instantiation, mapping and update

| Argument | Meaning / default |
| --- | --- |
| `-t, --task` | Required: `instantiate`, `map`, `all`, `update` |
| `--config, --map-config` | Instantiation/mapping configuration; `instantiation_config.json` |
| `--construct-config` | Connection/model configuration; `construct_config.json` |
| `-ldb, --logic_database` | Abstract source; `instantiation.logic_database` (supplied `DOELG-abstract`) |
| `-idb, --ins_database` | Instance destination, or existing instance for mapping; root `database` |
| `--events` | Initial-event GeoJSON; `instantiation.events_file`, falling back to `region`; update uses `update.events_file` |
| `--max-depth` | Propagation depth; supplied `2`; `0` permits initial events only |
| `-w, --workers` | Reserved; reference inference remains sequential |
| `--source-database` | For `update` only: baseline instance to copy; `update.source_database`, falling back to root `database` |
| `--snapshot` | For mapping: offline source snapshot instead of Neo4j |
| `--output-dir` | Mapping destination, or empty update-run directory; configured default otherwise |
| `--no-plots` | Keep GeoTIFFs and skip map PNGs |
| `--show-config` | Print effective settings and exit without connections/writes |

Menu 3 prompts for configuration and connection configuration first. `instantiate`/`all` then ask for source logic, initial events and instance database. `map` asks for an optional snapshot; a supplied snapshot replaces the database prompt. `update` asks for baseline database, logic database, new events and destination. Mapping tasks then ask for output directory and PNG choice. Depth/workers are configured in JSON or by direct CLI.

```powershell
python main/main_instantiation.py -t all -ldb DOELG-abstract -idb my-instance --output-dir data/ins_figures/my-instance
python main/main_instantiation.py -t map --snapshot path/to/snapshot.json --output-dir data/ins_figures/offline
python main/main_instantiation.py -t update --source-database my-instance -idb my-instance-update --events data/geo_dataset/events_zhengzhou/update.geojson
python main/main_instantiation.py -t instantiate --show-config
```

Mapping requires an absent/empty output directory. Update requires three different database names: baseline, abstract logic and update destination. Paths inside the instantiation configuration resolve from that file's directory; explicit CLI paths resolve from the project root.

### Extraction validation

| Argument | Meaning / default |
| --- | --- |
| `-t, --task` | Required: `evaluate`, `plot`, `table`, `prepare` |
| `-k, --kind` | `event`, `geo`, `relation`; required except for `table` |
| `--input` | Workbook; required for `prepare`/`evaluate` |
| `--output` | Required new workbook for `prepare`; CSV for `table` (default `<base>/results/table4.csv`) |
| `--only-raw` | Prepare predictions without ground truth; event/relation preparation only |
| `--base-dir` | For `plot`: one target's directory; default `data/extraction_validation_dataset/<kind>` (`relation` uses `event_relation`). For `table`: dataset parent, default `data/extraction_validation_dataset` |
| `--output-dir` | Plot destination; default `data/extraction_validation_dataset/results/<kind>` |
| `--shots` | Few-shot counts for plot/table; default `1 5 10 30` |

The menu asks for the target except on `table`, then shows only the applicable paths/options.

```powershell
python main/main_extraction_validation.py -t table
python main/main_extraction_validation.py -t evaluate -k event --input data/extraction_validation_dataset/event/default-10shot.xlsx
python main/main_extraction_validation.py -t plot -k geo --shots 1 5 10 30
python main/main_extraction_validation.py -t prepare -k event --input input.xlsx --output prepared.xlsx
```

### OELG, comparison, patterns and reproduction

Task aliases `patterns` and `reproduce` remain accepted for `pattern` and `reproduction`.

| Argument | Meaning / applicability |
| --- | --- |
| `-t, --task` | Required: `instantiate`, `map`, `compare`, `pattern`, `reproduction` |
| `--config` | Task/source configuration; defaults in the next table |
| `--instance-database` | OELG `instantiate`/`map` only; overrides `instance_database` |
| `--source` | `database` or `archive`: compare defaults to database; pattern defaults to archive in direct CLI and offers a choice in the menu; reproduction is archive-only |
| `--construct-config` | Connection override for database compare/pattern only |
| `--doelg-database`, `--oelg-database` | Instance graphs for database compare |
| `--doelg-raster-dir`, `--oelg-raster-dir` | Map roots to reuse in database compare; missing sets are generated |
| `--oelg-snapshot` | Advanced database-compare override; otherwise automatic discovery/metadata; no menu prompt |
| `--pattern-database` | Generalized logic graph for database pattern statistics |
| `--output-dir` | OELG audit directory (`instantiate`), OELG map directory (`map`), fresh database-validation directory, or reproduction directory |
| `--no-plots` | OELG `instantiate`: skip mapping entirely. `map`, database compare and reproduction: retain GeoTIFFs, skip map PNGs; comparison charts remain enabled |
| `--overwrite` | Replace comparison reports for compare/reproduction; database validation still requires an empty run directory |
| `--show-source` | For compare/pattern/reproduction: print source settings and exit without service connections |

| Task/source | Default configuration under `instantiation_validation/` | Restrictions |
| --- | --- | --- |
| 5.1 / 5.2 | `oelg_config.json` | No `--source`, `--construct-config`, `--show-source` or `--overwrite`; shared connection settings/environment are used |
| Database 5.3 / 5.4 | `database_validation_config.json` | Pattern uses the logic graph only; no instance/raster, PNG or overwrite options |
| Archive 5.3 | `importance_comparison_config.json` | Configure raster/output paths in JSON; no `--output-dir` or `--no-plots` |
| Archive 5.4 | `paper_experiment_config.json` | Configuration/source-display only; no output/PNG/overwrite options |
| 5.5 | `paper_experiment_config.json` | Archive only; rejects database options |

The 5.1/5.2 menu asks for configuration, instance database, output and map choice. Database 5.3 asks for configuration, connection configuration, both instance databases, both raster roots, output and PNG choice. Database 5.4 asks for configuration, connection configuration, logic database and output. Archive 5.4 needs only its configuration; 5.5 also offers output, PNG and report-replacement choices.

```powershell
python main/main_validation.py -t instantiate --instance-database my-oelg
python main/main_validation.py -t map --instance-database my-oelg
python main/main_validation.py -t compare --doelg-database my-instance --oelg-database my-oelg
python main/main_validation.py -t compare --show-source
python main/main_validation.py -t pattern --source database --pattern-database DOELG-abstract
python main/main_validation.py -t reproduction
python main/main_validation.py -t reproduction --overwrite
python main/main_validation.py -t compare --source archive
```

Run archive compare after reproduction has produced its configured rasters. OELG map-only uses the snapshot selected in `oelg_config.json`; if instantiation printed a suffixed result name, select that result in the configuration before redrawing. Database compare discovers its result independently. Validation configuration paths resolve from their file's directory.

## Configuration and outputs

### Configuration files

| File | Main settings |
| --- | --- |
| `construct_config.json` | `neo4j` connection/credentials; `llm` models/API settings; `embeddings` Ollama addresses/models; `extraction` input/database/workers; `fusion`; `geocoding` endpoints/type graph/basin; `sensor_manual`; `factor_mapping`; `generalization` database, weights/threshold, rules, statistics and export paths; `translation` dictionary |
| `instantiation_config.json` | Root `database`, feature layers and mapping settings; `instantiation` inference settings; `update` baseline/destination/events/output/batch size |
| `instantiation_validation/oelg_config.json` | `abstract_database` (OELG causal source), `observation_database` (DOELG observation source), `instance_database`, `snapshot`, scenario layers, depth, environment filtering, audit output |
| `instantiation_validation/oelg_distance_config.json` | Shared `map_config`, OELG output directory and depth-only `depth_decay` (supplied `0.5`) |
| `instantiation_validation/database_validation_config.json` | Owning configuration paths; non-null database/raster overrides; default evaluation output |
| `instantiation_validation/importance_comparison_config.json` | Raster pairs, validation point sets, evaluation settings and comparison output |
| `instantiation_validation/paper_experiment_config.json` | Frozen snapshots, provenance/reference files, expected counts and reproduction output |

The main inference settings are `max_depth` (`2`), `study_area` (EPSG:4326 bounds), `elevation` (`120 m`), `ori_importance` (`120`), `importance_punish` (`30`), `i_fac` (`1`), `confidence_to_probability_scale` (`7`), `min_secondary_confidence`, distance bands/weights/search limits, and `basin_file` (`null` disables basin matching). Runtime settings are loaded before inference. Menu 3's source and destination overrides apply to both inference and mapping as appropriate.

Mapping uses root `region`, `features_folder`, `reference_raster` (or `grid_size`, default `200`), `distance_crs`, `default_cutoff_m`, per-event/type/geometry cutoff overrides, `geometry_kernels`, `spatial_aggregation`, `overlap_strength`, `path_boost`, hazard/type weights and `exclude_geo_types`. `initial_event_importance` controls mapped initial-event scores separately from inference's `ori_importance`. PNG/display settings and `target_coord` control map presentation and the cell-factor histogram. Review region, clipping bounds, initial events, feature/basin layers and metric CRS together when changing scenario.

The supplied OELG profile uses `max_depth=2`, `environment_mode=strict`, OELG environment rules, elevation `120 m` and `report_and_skip` for missing environment rules/layers. Database comparison derives defaults from the owning configurations, applies non-null profile overrides, then explicit CLI overrides. Use matched scenario inputs, propagation depth and grids for a live method comparison. Frozen reproduction settings remain independent of live settings.

For direct commands, Neo4j clients honor `NEO4J_URI`, `NEO4J_DATABASE_URL`, `NEO4J_USERNAME` and `NEO4J_PASSWORD`. An LLM entry's `api_key_env` overrides its configured key when populated; interactive startup applies the entered DeepSeek Key to DeepSeek clients throughout that session.

### Outputs and reruns

| Workflow | Default output | Rerun behavior |
| --- | --- | --- |
| Generalization | Selected graph; `all-cluster.pickle`; `data/generalization` | Shared export paths can be replaced by another run |
| DOELG instantiation | Instance graph; `data/logs/instantiation_execution.json`; configured event-tree/feature exports | Existing database is not cleared; choose a new database for an independent run |
| DOELG mapping | `data/ins_figures/distance_fields` | Output must be absent/empty |
| Update | New run under `data/ins_figures/updates`, including `update_status.json` and `maps/` | Baseline preserved; destination graph/output must be empty |
| OELG | Audit under `data/oelg_comparison/zhengzhou`; maps under `data/ins_figures/oelg_distance_fields` | Occupied result names/folders receive suffixes; retain the printed paths |
| Extraction validation | `data/extraction_validation_dataset/results` | `prepare` requires a new workbook |
| Database validation | New run under `data/database_validation`; `validation_status.json`, raster provenance and HTML/CSV comparison reports | Explicit run directory must be empty; valid maps/cache are reused |
| Paper reproduction | `data/paper_reproduction`; `reproduction_status.json`, maps and comparison reports | Matching maps reused; `--overwrite` replaces comparison reports |

The instantiation execution report is a summary, not a full console transcript. Record console output separately when a complete log is needed. Frozen inputs are under `data/previous_paper_experiment`; geometry/scenario layers under `data/geo_dataset`; validation points under `data/validation_points`; dictionaries/appendix rules under `data/dictionary`.

### Auxiliary utilities

`main/main_geo_types.py` supports the Appendix A/B type graph used in Section 3.3.2; it is not a numbered menu task. An imported, valid `Geo-type` graph needs no rebuild.

```powershell
python main/main_geo_types.py -t preview
python main/main_geo_types.py -t validate -db Geo-type
```

Arguments: required `-t/--task` (`preview`, `validate`, `rebuild`), `--config` (construction configuration), `--rules` (default `geocoding.type_rules`), `-db/--database` (default `geocoding.type_database`) and `--output-dir` (default `data/geo_type_graph`). Preview is offline; validate reads the graph. The rebuild helper accepts only the legacy name `geos-type-1`, backs up/replaces that graph and validates it. Select the rebuilt name in `geocoding.type_database` before use; rerun the spatial stage to refresh existing extracted paths.

To update a sharing archive:

```powershell
python main/build_windows_package.py --output dist/DOELG_windows.zip
```

`--output` defaults to the path shown. `--prepare-data` instead restores only missing inputs from `DOELG_dataset.tar` and exits. Packaging includes current code, README and required inputs, encrypts shared credentials, and leaves source configuration values unchanged. Generated results, `.runtime` and the local-only `run_doleg.bat` are excluded. See [instantiation_validation/README.md](instantiation_validation/README.md) for additional validation details.

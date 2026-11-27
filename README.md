# DOELG task guide

DOELG constructs disaster–observation event logic from text, instantiates that knowledge for a regional disaster scenario, maps spatial observation requirements, and evaluates the results. Task numbers below follow `run.py`: **3.4** means main menu 3, subtask 4.

- [Start the program](#start-the-program)
- [Tasks and paper correspondence](#tasks-and-paper-correspondence)
- [Execution order and dependencies](#execution-order-and-dependencies)
- [Inputs before each task](#inputs-before-each-task)

## Start the program

On Windows x64, extract `DOELG_windows.zip` and double-click `Start_DOELG.bat`. The first launch downloads Python 3.13 and installs the locked dependencies; subsequent launches reuse `.runtime`. Keep the project in a writable folder. Internet access is required for initial setup. Setup errors are recorded in `.runtime/setup.log`.

Before the menu, enter the Neo4j server URI, username, password and DeepSeek API Key. Press Enter to retain an existing value; password/Key input is hidden. Service settings are saved to `construct_config.json` and passed to all tasks in that menu session. Task-specific parameters apply only to that invocation. The launcher does not start Neo4j, Ollama, Nominatim or an LLM service.

The shared configuration stores passwords and API keys encrypted. Keep `.doelg-credentials.key` beside it. Automatic defaults require this bundled key, so the protection hides JSON plaintext but cannot prevent a package recipient from extracting credentials.

Use `Start_DOELG.bat --check` to prepare/check the environment without opening the menu. `python run.py --dry-run` previews task commands and does not save service settings or run tasks. Enter `0` to go back/exit; Ctrl+C cancels input or a running task. Blank optional task parameters use the selected script/configuration defaults. Required input paths must exist; preparation requires a new output workbook. The task input questions are listed below in the order shown by the launcher.

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

## Inputs before each task

These are the questions shown in the command window after starting `Start_DOELG.bat`, in their actual order. Type values only; do not type Python commands or argument names. All paths below are relative to the project root unless absolute paths are entered.

Press Enter at an optional question to use the displayed/configured default. For `y/n`, Enter means **No**. Worker counts must be positive integers; few-shot counts are space-separated positive integers. At a parameter question, `0` is a value, not Back. After the final answer, the task starts immediately.

### Startup settings

These four questions appear once, before the main menu:

| Order / displayed question | What to enter | Enter without typing |
| --- | --- | --- |
| 1. `Neo4j server URI` | Server address, e.g. `bolt://localhost:7687`; this is not a database name | Keep the existing address |
| 2. `Neo4j username` | Neo4j login username | Keep the existing username |
| 3. `Neo4j password` | Neo4j login password; hidden input | Keep the existing password; required if missing |
| 4. `DeepSeek API Key` | Key for the configured DeepSeek models; hidden input | Keep the existing Key; required if missing |

These values are saved; the task inputs below are not saved to configuration files. Startup collects service settings even for archive tasks; archive processing itself uses local files.

### Menus 1 and 2

All tasks **1.1–1.6** and **2.1–2.5** ask the following first:

| Order / displayed question | Meaning | Default |
| --- | --- | --- |
| 1. `Construction configuration` | JSON file containing connection, model, data and processing settings | `construct_config.json` |
| 2. `Neo4j database` | History database to write/enrich, or existing history database to generalize | Selected file's `extraction.database` for menu 1; `generalization.database` for menu 2 |
| 3. `Worker count` | Number of processing workers | Corresponding section's `workers` |

Only **1.1 extract** and **1.6 all** then ask **4. `Input JSON directory`**: the folder of corpus JSON files, defaulting to `extraction.input_dir` (supplied `data/extraction_source_dataset/sample_corpus`). The other tasks have no additional input questions.

### Menu 3

Every task starts with **1. `Instantiation and mapping configuration`** (default `instantiation_config.json`), then **2. `Connection and model configuration`** (default `construct_config.json`). The following questions depend on the task:

| Task | Remaining questions, in order |
| --- | --- |
| **3.1 instantiate** | `Source logic database` → `Initial event GeoJSON` → `Instance database` |
| **3.2 map** | `Offline source snapshot (optional)` → `Instance database` **only if the snapshot was left blank** → `New map output directory` → `Skip map PNGs? y/n` |
| **3.3 all** | `Source logic database` → `Initial event GeoJSON` → `Instance database` → `New map output directory` → `Skip map PNGs? y/n` |
| **3.4 update** | `Existing instance database to copy` → `Source logic database` → `New initial events GeoJSON` → `New update instance database` → `New update output directory` → `Skip map PNGs? y/n` |

| Question | Meaning / default when left blank |
| --- | --- |
| `Source logic database` | Abstract knowledge source; `instantiation.logic_database`, supplied `DOELG-abstract` |
| `Initial event GeoJSON` | Initial disaster polygons; `instantiation.events_file`, falling back to `region` |
| `Instance database` | Destination for instantiation, or existing graph for map-only; root `database` |
| `Offline source snapshot (optional)` | Exported JSON snapshot to map without querying Neo4j; blank means read Neo4j |
| `New map output directory` | Empty/absent map destination; blank uses root `output_dir` |
| `Existing instance database to copy` | Baseline graph preserved by update; `update.source_database`, falling back to root `database` |
| `New initial events GeoJSON` | Additional evidence for update; `update.events_file` |
| `New update instance database` | Separate empty destination; `update.instance_database` |
| `New update output directory` | Empty update-run directory; blank creates a new run below `update.output_dir` |
| `Skip map PNGs? y/n` | Yes keeps GeoTIFFs but omits map PNGs; No generates both |

Inference depth, scoring and grid settings are read from the selected JSON; the `.bat` menu does not ask for them individually.

### Menu 4

`Evaluation target` is a numbered selection: **1 event**, **2 geo**, **3 relation**. It is not shown for Table 4 export.

| Task | Questions, in order |
| --- | --- |
| **4.1 evaluate** | `Evaluation target` → `Input workbook` |
| **4.2 plot** | `Evaluation target` → `Workbook base directory` → `Few-shot sample counts` → `Plot output directory` |
| **4.3 table** | `Workbook base directory` → `Few-shot sample counts` → `Output CSV file` |
| **4.4 prepare** | `Evaluation target` → `Input workbook` → `New output workbook` → `Prepare predictions without ground truth? y/n` **for event/relation only** |

| Question | Meaning / default when left blank |
| --- | --- |
| `Input workbook` | Existing `.xlsx` to evaluate/prepare; required, cannot be blank |
| `New output workbook` | New `.xlsx` for prepared data; required and must not exist |
| `Workbook base directory` | For plot: one target's directory, default `data/extraction_validation_dataset/<kind>` (`relation` uses `event_relation`). For table: parent containing all three target folders, default `data/extraction_validation_dataset` |
| `Few-shot sample counts` | Workbook shot settings, e.g. `1 5 10 30`; this is not a worker count; default `1 5 10 30` |
| `Plot output directory` | Figure destination; default `data/extraction_validation_dataset/results/<kind>` |
| `Output CSV file` | Table 4 results file; default `<base-directory>/results/table4.csv` |
| `Prepare predictions without ground truth? y/n` | Yes prepares prediction columns only; No also processes ground-truth content |

### Menu 5

For **5.4**, first choose `Validation data source`: **1 Local Neo4j databases**, **2 Existing archives**. Task **5.3** uses database comparison automatically; **5.5** uses archives automatically.

| Task | Questions, in order |
| --- | --- |
| **5.1 instantiate** | `Validation configuration` → `OELG instance database` → `Output directory` → `Skip maps entirely? y/n` |
| **5.2 map** | `Validation configuration` → `OELG instance database` → `Output directory` → `Skip map PNGs? y/n` |
| **5.3 compare** | `Validation configuration` → `Connection configuration` → `DOELG instance database` → `OELG instance database` → `DOELG map root (reuse existing rasters)` → `OELG map root (reuse existing rasters)` → `New local evaluation output directory` → `Skip map PNGs? y/n` |
| **5.4 pattern — database** | `Validation data source` → `Validation configuration` → `Connection configuration` → `Generalized logic database` → `New local evaluation output directory` |
| **5.4 pattern — archive** | `Validation data source` → `Validation configuration` |
| **5.5 reproduction** | `Validation configuration` → `Output directory` → `Skip map PNGs? y/n` → `Replace existing comparison reports? y/n` |

`Validation configuration` selects the task's JSON file. Its default is under `instantiation_validation/`: `oelg_config.json` for 5.1/5.2, `database_validation_config.json` for 5.3/database 5.4, and `paper_experiment_config.json` for archive 5.4/5.5.

| Question | Meaning / default when left blank |
| --- | --- |
| `Connection configuration` | Connection-settings JSON; default comes from the database-validation profile, normally root `construct_config.json` |
| `DOELG instance database` / `OELG instance database` | Existing result graphs to compare; 5.1 instead writes the selected OELG destination. Defaults come from the owning instantiation configurations/profile |
| `DOELG map root (reuse existing rasters)` / `OELG map root (reuse existing rasters)` | Each method's map root, containing `output_event_maps_grid/geotiff` and `output_factor_maps_grid_geotiff`; default comes from the profile/map settings. Missing raster sets are generated from the selected existing graphs |
| `Generalized logic database` | Abstract graph for pattern statistics, not an instance graph; derived from the profile/construction configuration |
| `New local evaluation output directory` | Empty result directory for 5.3/database 5.4; blank creates a fresh run under the profile's `output_dir`, supplied `data/database_validation` |
| `Output directory` | 5.1: OELG audit exports, default `output_folder`; 5.2: OELG maps, default from distance-map settings; 5.5: reproduction outputs, default from the paper profile |
| `Skip maps entirely? y/n` | 5.1 only: Yes writes graph/audit results and skips all mapping; No also generates maps |
| `Skip map PNGs? y/n` | Yes keeps map GeoTIFFs and omits map PNGs; comparison curves/reports are still produced |
| `Replace existing comparison reports? y/n` | 5.5 only: Yes permits replacement of comparison reports; No preserves existing reports |

## Execution order and dependencies

### Choose a workflow

| Goal | Order | Requirements                                                                                        |
| --- | --- |-----------------------------------------------------------------------------------------------------|
| Verify archived paper values | **5.5** directly; **4.3** for Table 4 only; **5.4 → Existing archives** for patterns only | need Neo4j instance connection                                                                      |
| Infer/map using supplied abstract graphs | Import dumps → **3.3** (or **3.1 → 3.2**) | `DOELG-abstract`, local scenario inputs and a separate instance destination                         |
| Construct new knowledge from text | **1.6 → 2.5 → 3.3** | Use the same history database for menus 1–2; select it explicitly as menu 3's source logic database |
| Compare live methods | **3.3** and **5.1** → **5.3** | Same scenario/depth/grid; populated instance graphs or complete compatible rasters for both methods |
| Map OELG separately | **5.1 → 5.2** | Needed only if maps were skipped or must be redrawn; 5.3 can also generate missing maps             |
| Incrementally update a scenario | Baseline **3.1/3.3 → 3.4** | Existing baseline, compatible abstract logic, new events, separate empty update database            |
| Evaluate new extraction workbooks | **4.4 → 4.1/4.2/4.3** | Verified ground truth; skip 4.4 for already prepared workbooks                                      |
| Inspect live abstract patterns | **2.1** (normally **2.5**) → **5.4 → Local Neo4j databases** | Select the generalized logic database, not an instance database                                     |

Each `all` covers only its own menu group. Selecting a database in one group does not select it automatically in another. For separate stage runs, 1.2–1.5 require extracted records; 2.2 requires the hierarchy and cluster file from 2.1; 2.3 requires abstract disasters plus extracted observations/factors. Complete observation-pattern and spatial stages before full generalization. Run 2.4 after 2.3 for observation coverage statistics. Keep cluster files and graph versions matched; use separate configured export paths for independent datasets.

The default extraction input is the small `sample_corpus`, intended for demonstration. It does not reproduce the full paper experiment or guarantee its graph counts. The larger bundled corpus is `wiki_corpus`; explicitly select and verify the intended input for a full run. Live results depend on corpus, graph/model versions and settings. Task 5.5 verifies frozen paper values: 24 Table 4 values, 80 field metrics and the archived pattern summary. It does not rebuild historical graphs or perform Figure 22's update. Tasks generate graph content/maps/reports; manuscript screenshot layouts and formatted tables may need separate composition.

### Databases and services

Use Neo4j 5.24.0 or later with APOC. For supplied knowledge, import `Geo-type.dump`, `DOELG-abstract.dump` and `OELG-abstract.dump` into the same Neo4j instance under the corresponding names. `geocoding.type_database` selects the type graph (supplied `Geo-type`). 
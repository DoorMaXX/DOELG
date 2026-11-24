# KGC_main command-line guide

## Interactive console

On Windows, double-click `run_kgc.bat` to open a plain text menu using the `KGC` conda environment. The launcher detects Conda from the active environment, PATH, common installation directories, or the user's registered Conda environments. If it cannot locate Conda, use Anaconda Prompt:

```powershell
conda activate KGC
python run.py
```

The launcher introduces the project, shows the current project directory and Python interpreter, and presents the following menu. It does not initialize model clients or connect to Neo4j until a selected task is started.

| Main menu | Displayed group | Script called |
| --- | --- | --- |
| `1` | Extraction and fusion | `main/main_extraction.py` |
| `2` | Knowledge generalization | `main/main_generalization.py` |
| `3` | Instantiation and mapping | `main/main_instantiation.py` |
| `4` | Extraction validation | `main/main_extraction_validation.py` |
| `5` | Method validation and paper reproduction | `main/main_validation.py` |
| `0` | Back / Exit | Exit the launcher |

Choose a group, choose a task, and enter the applicable parameters. After the last parameter, the launcher prints the generated command and runs it immediately. Task output remains visible in the same console. When the task completes or fails, the main menu is displayed again; a nonzero process exit code is reported as a failed task.

### Input and navigation

- Enter `0` at a submenu to return to the main menu. At a parameter prompt, `0` is an input value, not a Back command.
- Press Enter at an optional parameter to omit that argument and let the called script use its default. A displayed default is explanatory; it is not saved into the configuration.
- Required input workbooks and new output workbooks cannot be left blank. A preparation output workbook must not already exist.
- For a `y/n` question, Enter, `n`, and `no` mean No; `y` and `yes` mean Yes. Answers are case-insensitive.
- Worker counts must be positive integers. Few-shot counts are positive integers separated by spaces, such as `1 5 10 30`.
- Supplied configuration files, input files, and input directories must exist. Relative paths resolve from the project root; paths containing spaces can be entered with or without surrounding double quotes. Output-directory constraints are checked by the called script.
- Ctrl+C cancels parameter entry or the running task and returns to the main menu. Closing standard input exits the launcher.

Each command runs from the project root using the same Python interpreter as `run.py`, with UTF-8 output enabled. Parameters affect that invocation only: selecting database `case` in menu 1 does not select it automatically in menu 2. Enter the intended database again at every relevant prompt. Configuration files are not edited, and services are not started automatically.

### Preview commands

```powershell
python run.py --dry-run
```

On Windows, `run_kgc.bat --dry-run` starts the same preview mode in the KGC environment. Menus, required fields, and path validation behave as usual, but the launcher only prints the generated command. No pipeline task is started.

### Menu 1: Extraction and fusion

| Task number | Task | Purpose |
| --- | --- | --- |
| `1` | `extract` | Extract geographic entities, events, and their relations |
| `2` | `fuse` | Merge duplicate entities and events in the graph |
| `3` | `env` | Extract hazard-forming environment descriptions |
| `4` | `obs_pat` | Extract sensor observation patterns |
| `5` | `spatial` | Geocode features and calculate spatial relationships |
| `6` | `all` | Run extraction, fusion, environment extraction, observation patterns, and spatial computation |

Prompts appear in this order:

| Prompt | Command argument | When shown | Enter uses |
| --- | --- | --- | --- |
| Construction configuration | `--config` | Every task | Root `construct_config.json` |
| Neo4j database | `--database` | Every task | `extraction.database` in the selected configuration |
| Worker count | `--workers` | Every task | `extraction.workers` |
| Input JSON directory | `--input_dir` | `extract`, `all` | `extraction.input_dir` |

Live tasks require the services used by their selected stage. In particular, fusion uses the configured embedding service, observation-pattern extraction may use the sensor-manual embedding model, and spatial computation uses geocoding services and the `geos-type-1` auxiliary database. These dependencies are described in the configuration and direct-command reference below.

### Menu 2: Knowledge generalization

| Task number | Task | Purpose |
| --- | --- | --- |
| `1` | `disaster_gen` | Cluster disaster pairs and construct abstract events |
| `2` | `spatial_rule` | Generalize spatial relationships and environment rules |
| `3` | `obs_gen` | Generalize observation events and standard factors |
| `4` | `stats` | Export graph statistics and observation summaries |
| `5` | `all` | Run all generalization stages in the order above |

Every task asks for Construction configuration (`--config`), Neo4j database (`--database`), and Worker count (`--workers`), in that order. Defaults come from root `construct_config.json`, `generalization.database`, and `generalization.workers`. Select the database containing the extracted graph; the launcher does not copy data from the database used in menu 1. `stats` reads graph data and writes local exports; the other tasks construct or update graph knowledge.

### Menu 3: Instantiation and mapping

| Task number | Task | Purpose |
| --- | --- | --- |
| `1` | `instantiate` | Instantiate regional disaster and observation events |
| `2` | `map` | Generate event and observation-factor distance fields |
| `3` | `all` | Run instantiation followed by mapping |

| Task | Prompts, in order |
| --- | --- |
| `instantiate` | Source logic database; Initial event GeoJSON; Instance database |
| `map` | Offline source snapshot; Instance database if no snapshot was supplied; Mapping configuration; New map output directory; Skip map PNGs? |
| `all` | Source logic database; Initial event GeoJSON; Instance database; Mapping configuration; New map output directory; Skip map PNGs? |

The source logic database defaults to `events-logic-all-1zz-sensor-en-0904`, the event file to `data/geo_dataset/events_zhengzhou/onset.geojson`, the instance database to `events-ins-all-zhengzhou-test-1`, and the mapping configuration to root `instantiation_config.json`. Supply a different source database if generalization was performed elsewhere. Source and instance databases serve different roles: instantiation reads the former and writes the latter.

For `map`, supplying a snapshot passes `--snapshot` and omits the instance-database prompt; mapping reads the snapshot. Leaving it blank uses Neo4j. An omitted output directory uses the mapping configuration's `output_dir`; mapping requires an empty or absent destination. Answering Yes to Skip map PNGs adds `--no-plots`, while GeoTIFFs are still generated. The launcher does not prompt for the script's `--workers` argument.

### Menu 4: Extraction validation

| Task number | Task | Prompts, in order |
| --- | --- | --- |
| `1` | `evaluate` | Evaluation target; Input workbook |
| `2` | `plot` | Evaluation target; Workbook base directory; Few-shot sample counts; Plot output directory |
| `3` | `table` | Workbook base directory; Few-shot sample counts; Output CSV file |
| `4` | `prepare` | Evaluation target; Input workbook; New output workbook; Prepare predictions without ground truth? for event/relation only |

The Evaluation target submenu is `1` = `event`, `2` = `geo`, `3` = `relation`, and `0` = Back. `table` has no target submenu because it evaluates all three targets.

For `plot`, the base directory holds one target's few-shot workbooks. The default is `data/extraction_validation_dataset/<kind>`; `relation` uses `event_relation`. For `table`, the base is the parent dataset directory containing the `event`, `geo`, and `event_relation` subdirectories; its default is `data/extraction_validation_dataset`. Both tasks default to shots `1 5 10 30`.

The default plot destination is `data/extraction_validation_dataset/results/<kind>`. The default Table 4 CSV is `<base-directory>/results/table4.csv`. Preparation requires a new output workbook and offers `--only-raw` for event/relation targets; this question is omitted for geographic validation. All four tasks operate on local workbooks. `prepare` parses existing extraction strings into structured JSON columns; it does not request new model extractions. Prepared files are then used by evaluation and plotting.

### Menu 5: Method validation and paper reproduction

| Task number | Task | Default configuration under `instantiation_validation` | Prompts after configuration |
| --- | --- | --- | --- |
| `1` | `reproduce` | `paper_experiment_config.json` | Output directory; Skip map PNGs?; Replace existing comparison reports? |
| `2` | `patterns` | `paper_experiment_config.json` | None |
| `3` | `compare` | `importance_comparison_config.json` | Replace existing comparison reports? |
| `4` | `instantiate` | `oelg_config.json` | Output directory; Skip maps entirely? |
| `5` | `map` | `oelg_config.json` | Output directory; Skip map PNGs? |

Every task first asks for Validation configuration (`--config`). An omitted output directory uses the selected workflow's configured destination. The directory means reproduction output for `reproduce`, audit output for OELG `instantiate`, and map output for OELG `map`.

The two skip questions both pass `--no-plots`, but have different effects: OELG `instantiate` skips the entire mapping stage, whereas `map` and `reproduce` skip map PNGs while retaining GeoTIFF output. Reproduction still produces comparison charts. Answering Yes to Replace existing comparison reports adds `--overwrite`; Enter leaves existing-output protection enabled.

`reproduce` and `patterns` use archived local inputs, and `compare` evaluates existing raster fields with local validation points. OELG `instantiate` and `map` use configured Neo4j graphs. Menu 5's `instantiate` constructs the OELG comparison baseline; menu 3's `instantiate` runs the main DOELG workflow.

### Typical menu workflows

For a newly constructed graph, run menu **1 → 6 (`all`)**, then **2 → 5 (`all`)** against the same database, for example `case`. For spatial inference, use **3 → 3 (`all`)**, enter `case` as Source logic database, and choose a separate instance database. Each `all` option covers its own group only; it does not start the next main-menu group. Individual tasks can be selected when only one stage needs to be rerun.

For an offline paper-results check, choose **4 → 3 (`table`)** or **5 → 1 (`reproduce`)**. If reproduction outputs already exist and their maps match the frozen inputs, those maps are reused; answer Yes to Replace existing comparison reports to regenerate the comparison outputs. Choose **5 → 2 (`patterns`)** to verify archived pattern counts only.

## Direct command-line reference

The existing scripts can also be called directly. Run the following commands from the **KGC_main project root** in PowerShell with the KGC conda environment:

```powershell
conda activate KGC
$env:PYTHONUTF8 = "1"
```

Use `python main/<filename> --help` to inspect each entry point. Replace placeholders in angle brackets with actual values. Database tasks require an accessible Neo4j database; some extraction and generalization stages also require the configured model service.

### Reproducing the paper results

```powershell
python main/main_extraction_validation.py -t table
python main/main_validation.py -t reproduce
python main/main_validation.py -t patterns
```

`table` exports the 24 Table 4 F1 values to `data/extraction_validation_dataset/results/table4.csv`. Event evaluation combines subject/object and argument matches into a micro-F1, filtering empty names and `None`. Geographic evaluation uses the migrated `KGC_2/data/extraction_dataset/geo/new-2` workbooks, checks relation words as well as unordered endpoints, and reports combined entity/relation micro-F1. `evaluate` and `plot` use these same calculators.

`reproduce` reads `instantiation_validation/paper_experiment_config.json`. It verifies migrated input hashes, reproduces Table 4, reconstructs DOELG and OELG distance fields from archived scores and factor associations, compares their rasters using local validation points, and checks the 80 metric values reported in Tables 6 and 7. It writes maps, comparison reports, and `reproduction_status.json` under `data/paper_reproduction`. Use `--output-dir PATH` for another result directory. On subsequent runs, matching maps are reused; pass `--overwrite` to replace generated comparison reports. `--no-plots` disables map PNGs; comparison charts are still produced.

This workflow does not contact Neo4j, Ollama, geocoding services, or an LLM. It replays the archived experiment inputs; it does not rebuild the historical graphs. DOELG retains the original 394 factor-to-event associations from `events-ins-all-zhengzhou-1-sensor-en-0904`. OELG retains the original strict-environment snapshot and depth-only scores. Source versions and SHA-256 hashes are recorded in `data/paper_experiment/provenance.json`. Historical paths embedded in snapshot metadata identify their origin and are not runtime dependencies because the geometries are included.

`patterns` verifies 1,440 event-pair groups containing 1,656 patterns in `data/paper_experiment/pattern_statistics.json`. The original generalization script identifies the source database as `events-logic-all-1zz-sensor-en-0904`; a database dump is not included. The existing construction and live-map configuration files remain separate from the archived paper profile.

### Construction configuration

`main_extraction.py`, `main_generalization.py`, and their supporting modules read `construct_config.json` in the project root. Use `--config <config.json>` to select a different complete construction configuration. The configuration is selected before model clients are initialized. Relative configuration and data paths are resolved from the project root, independently of the current working directory.

Explicit command-line arguments (`--database`, `--workers`, and extraction's `--input_dir`) override the corresponding entry-point defaults in the configuration. Module-specific worker limits, such as `fusion.workers` and `geocoding.encoding_workers`, are configured separately. Explicit arguments supplied directly to helper functions also override their configured defaults.

| Section | Settings |
| --- | --- |
| `neo4j` | Server URL, username, password, and default database for direct connector use |
| `llm` | Event and geographic extraction models, API addresses, temperatures, and credentials |
| `embeddings` | Ollama embedding models and optional server addresses |
| `extraction` | Default database, input directory, and worker count |
| `fusion` | DBSCAN parameters, cluster validation rounds, workers, and deduplication batch size |
| `geocoding` | Geocoding service endpoints, reference systems, basin data, workers, and path limits |
| `sensor_manual` | Sensor workbook, vector index directory, retrieval count, agent recursion limit, and evidence output |
| `factor_mapping` | Worker counts and minimum association score |
| `generalization` | Default database and workers, file paths, clustering weights and threshold, causal confidence parameters, environment elevation, observation resolution thresholds, and statistics limits |
| `translation` | Project-local Chinese-to-English graph vocabulary (`dictionary_path`) |
| `standalone` | Database names used by the supporting modules' standalone examples |

Configure the Neo4j server and credentials in the root `construct_config.json`:

```json
"neo4j": {
  "url": "bolt://localhost:7687",
  "username": "neo4j",
  "password": "<your-password>",
  "default_database": "case"
}
```

`url` is the server's Bolt address; use the actual host and port shown by your running Neo4j instance. `--database` selects a database on that server and does not change the host or port. The shared connector in `ontology/neo4jmodel.py` reads these settings for extraction, generalization, and instantiation. Live mapping and OELG validation also use these credentials without importing the graph models. Existing configurations with credentials embedded in `url` remain supported; explicit `username` and `password` fields take precedence. Mapping and OELG's direct-driver workflows retain their existing `NEO4J_DATABASE_URL`, `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` environment overrides.

In `llm.event` and `llm.geo`, `api_key_env` names an environment variable that overrides `api_key` when populated. The supplied configuration preserves the previous model and connection settings. For example, set `$env:DEEPSEEK_API_KEY` to supply model credentials through the environment.

Sensor manual retrieval uses `embeddings.sensor.batch_size` (default 16), `max_attempts` (3 total attempts per batch), `retry_delay_seconds` (2 seconds, increasing by the failed attempt number), and `timeout_seconds` (60 per HTTP request). Document and query embeddings share a request lock. Only transport failures and unavailable internal Ollama tokenizer/runner errors are retried; ordinary invalid requests and missing models are reported immediately.

The observation-pattern stage initializes the sensor index before submitting LLM workers. A verified index in `sensor_manual.index_dir` is reused when its workbook, model, and file hashes match; an old index without `sensor_index_manifest.json` is rebuilt once. Failed tasks are counted explicitly. The command exits with a failure status when any task remains incomplete; rerun `obs_pat` against the same database to process missing patterns. Existing model names, worker counts, and database selections are unchanged.

Geocoding requests use `geocoding.http`: `connect_timeout_seconds` (5), `read_timeout_seconds` (30), `max_attempts` (3 total attempts), `retry_delay_seconds` (1, increasing by the failed attempt number), and `lookup_batch_size` (50 or fewer IDs). Temporary transport failures, HTTP 429, and server errors are retried. Search and lookup failures return empty lists; reverse failures return an empty string. Spatial computation skips unavailable or incomplete geometry without inventing spatial metrics and reports computed, skipped, and missing-coordinate pair counts. A skipped eligible pair makes the command report an incomplete run after the remaining semantic/type paths have been processed.

Geometry insertion rejects empty or malformed OSM IDs and invalid coordinates. Existing empty-ID geometry links are ignored and can be geocoded again when `geocoding.do_encoding` is true; they are not deleted automatically. All existing geocoding endpoint addresses remain under `geocoding.search_url`, `lookup_url`, and `reverse_url`.

### Geographic type auxiliary graph

`geocoding.type_database` selects the spatial type-path database (default `geos-type-1`), and `geocoding.type_rules` points to `data/dictionary/geo_type_rules.json`. The catalog freezes Appendices A and B of `Appendix_A_to_D.docx`, including its SHA-256 and original table row numbers. It contains 78 distinct types and 126 relation rows, expanded into 401 unique directed `SPATIAL` relationships. Slash-separated endpoint groups are expanded; duplicate rules retain all source rows. Composite relation names are kept as printed. Appendix B row 85 prints a dash as its relation name; the stored name uses its explicit type, `Rely on`.

Names use the existing project vocabulary: `Earth Bank`, `In-Building Parking`, and `Telecom Line` retain the project's capitalization, and Appendix B's `Road` resolves to `Highway`. Printed appendix labels remain in provenance fields. Eight Appendix A types have no Appendix B edges: Cave, Coastline, Earth Bank, Sewer Vent, Sinkhole, Storage Tank, Valley, and Water Tower. They remain isolated rather than receiving inferred relationships. Type-path queries use the same aliases and continue to traverse `SPATIAL` edges without imposing a direction.

```powershell
python main/main_geo_types.py -t preview
python main/main_geo_types.py -t rebuild -db geos-type-1
python main/main_geo_types.py -t validate -db geos-type-1
```

| Argument | Purpose | Default |
| --- | --- | --- |
| `-t, --task` | Required: `preview`, `rebuild`, or `validate` | None |
| `--config` | Shared construction and connection configuration | Root `construct_config.json` |
| `--rules` | Frozen Appendix A/B catalog | `geocoding.type_rules` |
| `-db, --database` | Auxiliary type database | `geocoding.type_database` |
| `--output-dir` | Backup and validation report directory | `data/geo_type_graph` |

`preview` is offline. `rebuild` is restricted to `geos-type-1`: it saves and verifies a complete node/relationship JSON backup under `data/geo_type_graph`, replaces the graph in one transaction, validates it before committing, and writes a validation report. Other node labels or relationship types stop replacement. `validate` checks the live graph without writing to it. Connection credentials come from the shared construction configuration. Rebuilding this auxiliary graph does not rewrite existing `type_path` properties in extracted or generalized databases; rerun the spatial stage for the intended extraction database to refresh those paths.

```powershell
python main/main_extraction.py -t extract --config construct_config.json
python main/main_generalization.py -t disaster_gen --config construct_config.json
```

Graph categories are translated before lookup and insertion using `data/dictionary/dictionary.json`, copied from the original KGC_2 experiment. This covers geographic types and generic feature names, impact types, sensor types and platforms, standard factor names and related geographic types, observation weather/position categories, and structural and spatial relationship labels. Abstract event descriptions and cluster identifiers are constructed from the English categories. Matching uses the same English vocabulary, including `Administrative Region` and `Upstream-Downstream`; hyphens inside category names such as `In-Building Parking` remain intact. Construction and fusion use the `Impact` node class accepted by the ontology's disaster-argument relationships and used by the original graph.

Chinese extraction prompts remain available for the Chinese corpus. Source text, place names without dictionary entries, extracted impact surface names, raw factor names, and supporting evidence retain their original content, as in the original translation workflow. Dictionary lookup accepts comma/enumeration-mark and full-width/ASCII-parenthesis variants. Missing Chinese controlled terms and unlisted spatial predicates use the configured geographic model for translation. Successful translations and failures are cached for the current process; a failed translation returns an empty value and emits an English diagnostic instead of terminating construction. Geographic features without a type and relationships without valid endpoints/predicates are skipped individually. Disaster events without argument categories and observations without valid sensor arguments are skipped; later records continue. File summaries report written and skipped record counts. An unavailable dictionary file also uses the model fallback. Add recurring terms to the dictionary to make their translation fixed across runs. Restart the process to retry a cached failure after restoring the model service.

The usual extraction and generalization commands now produce the English graph directly; no post-construction translation script is required. These changes do not translate an existing database in place. To apply the new convention to an older Chinese graph, rerun construction into a selected destination database and regenerate its clustering and observation exports. Translating before semantic clustering may change embeddings and cluster assignments; the archived paper-reproduction profile remains a separate workflow.

### main_extraction.py: extraction and fusion

Cause recognition applies the class-level rules in Appendix E before writing each disaster-to-disaster `CAUSE` edge. The project includes the unchanged source workbook at `data/dictionary/Appendix_E.xlsx` and its runtime representation at `data/dictionary/causal_relation_rules.json`. The 194 active rules comprise 87 `Immediate` (`I`) and 107 `Altering` (`A`) pairs. Both `Confirmed` and `Preliminary` records are candidates; the three `Remove` records and blank matrix cells are excluded. Phenomenon matching tolerates case and whitespace, without guessing synonyms or using event surface descriptions as types.

Retained edges store `relation_type`, `relation_name`, and `affectType` as `Immediate` or `Altering`, plus `matrix_relation_code` and `matrix_review_status`. Evidence and occurrence counts are preserved. Reflexive edges are rejected; a same-type propagation candidate also requires different event nodes and different named geographic objects. Matrix membership supplies a class-level prior, while text evidence and subsequent spatial/environmental checks remain necessary. The `TRIGGER` relationship from disasters to observations is unaffected by the I/A classification.

When the extraction relation stage is rerun, existing extracted disaster `CAUSE` edges in the selected database are checked first: unsupported edges are removed and permitted edges are classified. The same check runs after fusion because merging can change event types or collapse distinct objects into one. Generalization also rejects unsupported type pairs and carries I/A properties onto `MergeDisasterEvent` and `GlobalDisasterEvent` edges. Instantiation uses the same rules and applies the paper's coefficient once: 1 for Immediate and 0.65 for Altering. Changing the extraction filter can change newly constructed graph statistics; archived paper reproduction continues to use its frozen snapshots.

```powershell
python main/main_extraction.py -t extract -i ./data/extraction_source_dataset/sample_corpus -db test-1 -w 20
python main/main_extraction.py -t fuse -db test-1
python main/main_extraction.py -t all -db test-1
```

| Argument | Description | Default |
| --- | --- | --- |
| `--config` | Construction configuration file | `construct_config.json` in the project root |
| `-t, --task` | Required: `extract`, `fuse`, `env` for environment descriptions, `obs_pat` for observation patterns, `spatial`, or `all` | None |
| `-i, --input_dir` | Directory containing input JSON files; used by `extract` and `all` | `extraction.input_dir` |
| `-w, --workers` | Number of workers for applicable stages | `extraction.workers` |
| `-db, --database` | Neo4j database name | `extraction.database` |

### main_generalization.py: knowledge generalization

```powershell
python main/main_generalization.py -t disaster_gen -db events-logic-all -w 20
python main/main_generalization.py -t spatial_rule -db events-logic-all
python main/main_generalization.py -t all -db events-logic-all
```

| Argument | Description | Default |
| --- | --- | --- |
| `--config` | Construction configuration file | `construct_config.json` in the project root |
| `-t, --task` | Required: `disaster_gen`, `spatial_rule`, `obs_gen`, `stats`, or `all` | None |
| `-w, --workers` | Number of workers for applicable stages | `generalization.workers` |
| `-db, --database` | Neo4j database name | `generalization.database` |

When run separately, `spatial_rule` loads the file selected by `generalization.paths.clusters`. Run `disaster_gen` first if those results are unavailable. Observation generalization writes and reads the same `generalization.paths.standard_factor_observations` file.

### main_instantiation.py: instantiation and mapping

```powershell
python main/main_instantiation.py -t instantiate -ldb events-logic-all-1zz-sensor-en-0904 -idb events-ins-all-zhengzhou-1-sensor-en-1030
python main/main_instantiation.py -t map -idb events-ins-all-zhengzhou-1-sensor-en-1030
python main/main_instantiation.py -t map --snapshot <snapshot.json> --output-dir <new-output-directory>
python main/main_instantiation.py -t all -idb <instance-database>
```

| Argument | Description | Default |
| --- | --- | --- |
| `-t, --task` | Required: `instantiate`, `map`, or `all` | None |
| `-ldb, --logic_database` | Source logic database | `events-logic-all-1zz-sensor-en-0904` |
| `-idb, --ins_database` | Instance database for writing or mapping; overrides `database` in the map configuration | `events-ins-all-zhengzhou-test-1` |
| `-w, --workers` | Accepted worker count; the current sequential instantiation implementation does not use it to control concurrency | `10` |
| `--events` | Initial event GeoJSON file | `data/geo_dataset/events_zhengzhou/onset.geojson` |
| `--map-config` | Mapping configuration file | `instantiation_config.json` in the project root |
| `--snapshot` | Read an existing snapshot instead of querying the database for mapping | Not set |
| `--output-dir` | Mapping output directory; must be empty or absent | `output_dir` in the mapping configuration |
| `--no-plots` | Export GeoTIFFs without PNGs | Off |

Instantiation uses the Zhengzhou region and maximum depth of 2 defined in the entry point. It writes to the instance database. Mapping refuses to overwrite a nonempty output directory.

### main_extraction_validation.py: extraction validation

```powershell
python main/main_extraction_validation.py -t evaluate -k event --input data/extraction_validation_dataset/event/default-10shot.xlsx
python main/main_extraction_validation.py -t evaluate -k geo --input data/extraction_validation_dataset/geo/default-10shot.xlsx
python main/main_extraction_validation.py -t evaluate -k relation --input data/extraction_validation_dataset/event_relation/default-10shot.xlsx
python main/main_extraction_validation.py -t plot -k geo --shots 1 5 10 30
python main/main_extraction_validation.py -t prepare -k event --input <input.xlsx> --output <new-output.xlsx>
```

| Argument | Description | Default |
| --- | --- | --- |
| `-t, --task` | Required: `prepare`, `evaluate`, `plot`, or `table` | None |
| `-k, --kind` | Required except for `table`: `event`, `geo`, or `relation` | None |
| `--input` | Input file for `prepare` and `evaluate` | Required for those tasks |
| `--output` | New output workbook for `prepare`, or result CSV for `table` | Required for `prepare`; `results/table4.csv` for `table` |
| `--only-raw` | Prepare predictions without ground truth; supported for event and relation datasets | Off |
| `--base-dir` | Directory of few-shot files for plotting | `data/extraction_validation_dataset/<kind>`; relation uses `event_relation` |
| `--output-dir` | Plot output directory | `data/extraction_validation_dataset/results/<kind>` |
| `--shots` | Few-shot sample counts for plotting | `1 5 10 30` |

Evaluation expects `raw_json` and `true_json` columns. Preparation retains the existing fallback from missing ground truth to predictions; check missing annotations before interpreting scores.

### main_validation.py: instantiation method comparison

```powershell
python main/main_validation.py -t instantiate
python main/main_validation.py -t map
python main/main_validation.py -t reproduce
python main/main_validation.py -t compare --overwrite
python main/main_validation.py -t compare --config <comparison-config.json>
```

| Argument | Description | Default |
| --- | --- | --- |
| `-t, --task` | Required: `instantiate`, `map`, `compare`, `reproduce`, or `patterns` | None |
| `--config` | Configuration for the selected task | `oelg_config.json` for live OELG tasks; `importance_comparison_config.json` for comparison; `paper_experiment_config.json` for reproduction and patterns, all under `instantiation_validation` |
| `--no-plots` | Skip mapping for `instantiate`, or export rasters without PNGs for `map` | Off |
| `--output-dir` | Audit output for `instantiate`, maps for `map`, or result directory for `reproduce` | The configured directory |
| `--overwrite` | Allow comparison reports to be replaced for `compare` or `reproduce` | Off |

OELG mapping also reads `instantiation_validation/oelg_distance_config.json` and its shared map configuration. Instantiation writes a new snapshot to the configured instance database. If the snapshot or output directory is occupied, a suffix is added; update the comparison configuration to point to the actual output. See `instantiation_validation/README.md` for details.

The subsidence validation set is named `Subsidence` and uses `data/validation_points/Subsidence/Subsidence.shp` with matching `.shx`, `.dbf`, `.prj`, and `.cpg` files. Comparison tables, figures, and exported filenames use this name. Historical `collapse` point-set identifiers are accepted as aliases; they do not refer to the distinct `Structural Collapse` hazard. Renaming the layer does not change its points or evaluation values.

### Data layout

Directory names describe their role; runtime exports use descriptive filenames without trial suffixes such as `_1`, `_07`, or `_final`. Geographic feature filenames and validation workbooks retain their category names and few-shot labels because loaders use them as identifiers. Corpus document titles remain unchanged.

| Directory | Contents |
| --- | --- |
| `data/dictionary` | Translation and causal/spatial rules, with the source appendix workbook |
| `data/extraction_source_dataset/wiki_corpus` | Full Wikipedia and reference-text corpus |
| `data/extraction_source_dataset/sample_corpus` | Default small extraction corpus, including `observation_samples.json` |
| `data/extraction_validation_dataset` | Event, geographic, and relation evaluation inputs and current figures |
| `data/generalization` | `standard_factor_observations.json`, `pattern_statistics.json`, and `disaster_observations.json` exported by live generalization |
| `data/geo_dataset` | Zhengzhou feature geometry, initial events, and the two Henan basin layers used by the code |
| `data/geo_type_graph` | Spatial type-graph backup and validation report |
| `data/ins_figures/distance_fields` | Current main-method distance fields and their complete source snapshot |
| `data/logs` | Instantiation log and `instantiation_execution.json` |
| `data/paper_experiment` | Frozen paper inputs, including `pattern_statistics.json`, snapshots, references, and provenance |
| `data/paper_reproduction` | Reconstructed maps, comparisons, and verification results |
| `data/sensor_dataset` | Sensor reference workbook |
| `data/validation_points` | Disaster validation point layers |

`data/cleanup_manifest.json` records the removed redundant files, retained equivalents, and renamed paths. Unused DEMs, broad-area basin layers, and old evaluation materials were removed only after verifying identical originals in KGC_2. The current project does not load those external originals. Duplicate trial map exports and the temporary pre-rename comparison backup were also removed. Live generalization statistics are separate from the frozen paper statistics.

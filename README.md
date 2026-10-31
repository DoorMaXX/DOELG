# Main command-line entry points

Run the following commands from the **KGC_main project root** in PowerShell with the KGC conda environment:

```powershell
conda activate KGC
$env:PYTHONUTF8 = "1"
```

Use `python main/<filename> --help` to inspect each entry point. Replace placeholders in angle brackets with actual values. Database tasks require an accessible Neo4j database; some extraction and generalization stages also require the configured model service.

## main_extraction.py: extraction and fusion

```powershell
python main/main_extraction.py -t extract -i ./data/extraction_source_dataset/originText-wiki-1 -db test-1 -w 20
python main/main_extraction.py -t fuse -db test-1
python main/main_extraction.py -t all -db test-1
```

| Argument | Description | Default |
| --- | --- | --- |
| `-t, --task` | Required: `extract`, `fuse`, `env` for environment descriptions, `obs_pat` for observation patterns, `spatial`, or `all` | None |
| `-i, --input_dir` | Directory containing input JSON files; used by `extract` and `all` | `./data/extraction_source_dataset/originText-wiki-1` |
| `-w, --workers` | Number of workers for applicable stages | `20` |
| `-db, --database` | Neo4j database name | `test-1` |

## main_generalization.py: knowledge generalization

At the last check, this entry point could not start in the KGC environment because `src/generalization.py` imports the missing module `instantiation_o`. The commands below describe the arguments defined in the code; the import must be fixed before they can run.

```powershell
python main/main_generalization.py -t disaster_gen -db events-logic-all -w 20
python main/main_generalization.py -t spatial_rule -db events-logic-all
python main/main_generalization.py -t all -db events-logic-all
```

| Argument | Description | Default |
| --- | --- | --- |
| `-t, --task` | Required: `disaster_gen`, `spatial_rule`, `obs_gen`, `stats`, or `all` | None |
| `-w, --workers` | Number of workers for applicable stages | `20` |
| `-db, --database` | Neo4j database name | `events-logic-all` |

When run separately, `spatial_rule` attempts to load existing clustering results. Run `disaster_gen` first if those results are unavailable.

## main_instantiation.py: instantiation and mapping

```powershell
python main/main_instantiation.py -t instantiate -ldb events-logic-all-1zz-sensor-en -idb events-ins-all-zhengzhou-1-sensor-en-1030
python main/main_instantiation.py -t map -idb events-ins-all-zhengzhou-1-sensor-en-1030
python main/main_instantiation.py -t map --snapshot <snapshot.json> --output-dir <new-output-directory>
python main/main_instantiation.py -t all -idb <instance-database>
```

| Argument | Description | Default |
| --- | --- | --- |
| `-t, --task` | Required: `instantiate`, `map`, or `all` | None |
| `-ldb, --logic_database` | Source logic database | `events-logic-all-1zz-sensor-en` |
| `-idb, --ins_database` | Instance database for writing or mapping; overrides `database` in the map configuration | `events-ins-all-zhengzhou-test-1` |
| `-w, --workers` | Accepted worker count; the current sequential instantiation implementation does not use it to control concurrency | `10` |
| `--events` | Initial event GeoJSON file | `data/geo_dataset/events_zhengzhou/onset.geojson` |
| `--map-config` | Mapping configuration file | `instantiation_config.json` in the project root |
| `--snapshot` | Read an existing snapshot instead of querying the database for mapping | Not set |
| `--output-dir` | Mapping output directory; must be empty or absent | `output_dir` in the mapping configuration |
| `--no-plots` | Export GeoTIFFs without PNGs | Off |

Instantiation uses the Zhengzhou region and maximum depth of 2 defined in the entry point. It writes to the instance database. Mapping refuses to overwrite a nonempty output directory.

## main_extraction_validation.py: extraction validation

```powershell
python main/main_extraction_validation.py -t evaluate -k event --input data/extraction_validation_dataset/event/default-10shot.xlsx
python main/main_extraction_validation.py -t evaluate -k geo --input data/extraction_validation_dataset/geo/default-10shot.xlsx
python main/main_extraction_validation.py -t evaluate -k relation --input data/extraction_validation_dataset/event_relation/default-10shot.xlsx
python main/main_extraction_validation.py -t plot -k geo --shots 1 5 10 30
python main/main_extraction_validation.py -t prepare -k event --input <input.xlsx> --output <new-output.xlsx>
```

| Argument | Description | Default |
| --- | --- | --- |
| `-t, --task` | Required: `prepare`, `evaluate`, or `plot` | None |
| `-k, --kind` | Required: `event`, `geo`, or `relation` | None |
| `--input` | Input file for `prepare` and `evaluate` | Required for those tasks |
| `--output` | New output workbook for `prepare`; it must not already exist or equal the input | Required for `prepare` |
| `--only-raw` | Prepare predictions without ground truth; supported for event and relation datasets | Off |
| `--base-dir` | Directory of few-shot files for plotting | `data/extraction_validation_dataset/<kind>`; relation uses `event_relation` |
| `--output-dir` | Plot output directory | `data/extraction_validation_dataset/results/<kind>` |
| `--shots` | Few-shot sample counts for plotting | `1 5 10 30` |

Evaluation expects `raw_json` and `true_json` columns. Preparation retains the existing fallback from missing ground truth to predictions; check missing annotations before interpreting scores.

## main_validation.py: instantiation method comparison

```powershell
python main/main_validation.py -t instantiate
python main/main_validation.py -t map
python main/main_validation.py -t compare
python main/main_validation.py -t compare --config <comparison-config.json>
```

| Argument | Description | Default |
| --- | --- | --- |
| `-t, --task` | Required: `instantiate` builds the OELG baseline and maps; `map` maps an existing graph; `compare` compares existing rasters | None |
| `--config` | Configuration for the selected task | `instantiation_validation/oelg_config.json` for instantiation and mapping; `instantiation_validation/importance_comparison_config.json` for comparison |
| `--no-plots` | Skip mapping for `instantiate`, or export rasters without PNGs for `map` | Off |
| `--output-dir` | Audit output for `instantiate`, or map output for `map` | The configured directory |
| `--overwrite` | Allow existing comparison outputs to be replaced; only valid for `compare` | Off |

OELG mapping also reads `instantiation_validation/oelg_distance_config.json` and its shared map configuration. Instantiation writes a new snapshot to the configured instance database. If the snapshot or output directory is occupied, a suffix is added; update the comparison configuration to point to the actual output. See `instantiation_validation/README.md` for details.

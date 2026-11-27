# Instance validation

To use the interactive launcher, run `python run.py` from the project root and choose main menu `5` (Method validation and paper reproduction). Its task numbers are `1` = OELG `instantiate`, `2` = OELG `map`, `3` = two-database `compare`, `4` = database/archive `pattern`, and `5` = archived paper `reproduction`. Main menu `3` runs the main DOELG workflow; its task `4` (`update`, Update and map) copies a baseline instance before adding new evidence and mapping the combined result. See the root README for the parameter prompts and defaults.

The paper experiment can be reproduced offline from the DOELG_main project root:

```powershell
python main/main_validation.py -t reproduce
python main/main_validation.py -t patterns
```

The reproduction profile is `paper_experiment_config.json`; its independent map settings are in `paper_distance_config.json`. Frozen graph-score snapshots and validation references are stored under `data/previous_paper_experiment`, and validation point layers are under `data/validation_points`. DOELG preserves the original 394 factor associations. OELG preserves the original strict-environment baseline snapshot. Outputs under `data/paper_reproduction` include reconstructed maps, Table 4, the Table 6/7 comparison, the 1,656-pattern summary, and `reproduction_status.json`. The script verifies 24 extraction values and 80 spatial evaluation values at manuscript precision. The pattern statistics contain 1,440 event-pair groups; the historical database dump is not bundled.

For a repeated run, use `-t reproduce --overwrite` to replace comparison reports while reusing maps with matching input signatures. Choose `--output-dir PATH` for a separate run. `--no-plots` disables map PNGs. These tasks replay archived experiment inputs and do not write to Neo4j.

Run the commands from the DOELG_main project root in the KGC environment:

```powershell
conda activate KGC
python main/main_validation.py -t instantiate
python main/main_validation.py -t compare
```

Task 5.1 creates the configured instance database in the running Neo4j instance if it is missing, and waits for it to be online before checking snapshot names or building results. Existing database contents are preserved. The default is `OELG-instance-{date}`, with `{date}` expanded to `YYYYMMDD` in Asia/Shanghai. Override it in the menu or with `--instance-database NAME`. Configuration templates are also resolved when mapping and comparing. Source knowledge databases and map-only destinations are not automatically created. Creation requires the appropriate Neo4j edition and database administration permissions.

The first command builds an OELG baseline, writes a new snapshot to the configured instance database, exports its audit files, and generates event and factor distance fields. To redraw maps from an existing configured snapshot, run `python main/main_validation.py -t map`. Use `--config PATH` to select an alternative configuration. For `instantiate`, `--no-plots` skips the entire mapping stage; for `map`, it skips map PNGs while retaining GeoTIFFs. `--output-dir PATH` selects the OELG audit directory for instantiation or the map directory for map-only runs.

Edit `oelg_config.json` for the abstract, observation, and instance databases and snapshot; edit `oelg_distance_config.json` for the OELG map output and `importance_comparison_config.json` for both methods' raster directories and validation points. Map settings shared with DOELG_main are read from `../instantiation_config.json`. Existing snapshots and nonempty OELG output folders are preserved by assigning a new suffix; when that happens, select the actual instance database and map root in task 5.3 (or update the archive comparison configuration for archive mode). Comparison outputs are protected from replacement unless `--overwrite` is supplied.

Task 5.5 (`reproduction`) is archive-only. Task 5.4 (`pattern`) offers database or archive inputs in the menu; direct commands default to archive. Task 5.3 (`compare`) defaults to comparing two existing instance databases, reusing each available raster set independently and generating only missing sets. For the previous offline comparison, run `python main/main_validation.py -t compare --source archive` after reproduction.

```powershell
python main/main_validation.py -t patterns --source database --pattern-database my-logic
python main/main_validation.py -t compare --doelg-database my-doelg --oelg-database my-oelg
python main/main_validation.py -t compare --doelg-raster-dir D:/maps/doelg --oelg-raster-dir D:/maps/oelg
python main/main_validation.py -t compare --show-source
```

`database_validation_config.json` resolves database names, snapshot and map roots from their owning construction/instantiation configurations unless non-null overrides are specified. CLI overrides take precedence. Use `--construct-config` for connection settings. `--show-source` reads configuration only and prints no credentials.

Map roots must contain event GeoTIFFs in `output_event_maps_grid/geotiff` and factor GeoTIFFs in `output_factor_maps_grid_geotiff`. A known database/snapshot mismatch prevents reuse. Metadata-free folders are marked unverified. Missing rasters are generated into a reusable cache under `data/database_validation/maps`; nonempty old outputs are preserved. Reused maps reflect their export time, so graph changes require fresh raster folders and removal of the relevant cache. New maps align to an available raster grid, while the evaluator verifies domain and point compatibility for each pair.

The task 5.3 menu has no snapshot-name prompt. Existing rasters can be used without querying Neo4j; their source metadata supplies the result name. If OELG mapping is needed, the latest complete, nonempty result with the supported schema is selected automatically from the chosen instance database, including names with a repeat-run suffix. This preflight runs before expensive DOELG mapping. An empty, unfinished or unsupported OELG result produces a database-specific message directing the user to task 5.1. Previous successful DOELG map exports are reused on retry. `--oelg-snapshot` is an optional advanced override only.

Database validation never writes graph data or creates instance graphs. Missing OELG fields require an existing complete graph snapshot from task 5.1. Pattern statistics require the generalized logic database with GlobalDisasterEvent/MergeDisasterEvent nodes. Each run writes a separate status report and raster provenance under `data/database_validation`; `--output-dir` selects an empty alternative. Current database results do not assert equality with the archived paper metrics.

The subsidence point-set identifier and layer are `Subsidence` and `../data/validation_points/Subsidence/Subsidence.shp`. All five Shapefile components share the `Subsidence` basename. The provenance manifest records their original source names and unchanged content hashes. Reference metrics use the same identifier, with all numeric values preserved. Historical `collapse` identifiers are normalized when reading older configurations or result tables; custom input paths are not rewritten.

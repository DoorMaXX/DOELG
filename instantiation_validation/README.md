# Instance validation

To use the interactive launcher, run `python run.py` from the project root and choose main menu `5` (Method validation and paper reproduction). Its task numbers are `1` = `reproduce`, `2` = `patterns`, `3` = `compare`, `4` = OELG `instantiate`, and `5` = OELG `map`. Main menu `3` runs the main DOELG instantiation and mapping workflow instead. See the root README for the parameter prompts and defaults.

The paper experiment can be reproduced offline from the KGC_main project root:

```powershell
python main/main_validation.py -t reproduce
python main/main_validation.py -t patterns
```

The reproduction profile is `paper_experiment_config.json`; its independent map settings are in `paper_distance_config.json`. Frozen graph-score snapshots and validation references are stored under `data/paper_experiment`, and validation point layers are under `data/validation_points`. DOELG preserves the original 394 factor associations. OELG preserves the original strict-environment baseline snapshot. Outputs under `data/paper_reproduction` include reconstructed maps, Table 4, the Table 6/7 comparison, the 1,656-pattern summary, and `reproduction_status.json`. The script verifies 24 extraction values and 80 spatial evaluation values at manuscript precision. The pattern statistics contain 1,440 event-pair groups; the historical database dump is not bundled.

For a repeated run, use `-t reproduce --overwrite` to replace comparison reports while reusing maps with matching input signatures. Choose `--output-dir PATH` for a separate run. `--no-plots` disables map PNGs. These tasks replay archived experiment inputs and do not write to Neo4j.

Run the commands from the KGC_main project root in the KGC environment:

```powershell
conda activate KGC
python main/main_validation.py -t instantiate
python main/main_validation.py -t compare
```

The first command builds an OELG baseline, writes a new snapshot to the configured instance database, exports its audit files, and generates event and factor distance fields. To redraw maps from an existing configured snapshot, run `python main/main_validation.py -t map`. Use `--config PATH` to select an alternative configuration. For `instantiate`, `--no-plots` skips the entire mapping stage; for `map`, it skips map PNGs while retaining GeoTIFFs. `--output-dir PATH` selects the OELG audit directory for instantiation or the map directory for map-only runs.

Edit `oelg_config.json` for the abstract, observation, and instance databases and snapshot; edit `oelg_distance_config.json` for the OELG map output and `importance_comparison_config.json` for both methods' raster directories and validation points. Map settings shared with KGC_main are read from `../instantiation_config.json`. Existing snapshots and nonempty OELG output folders are preserved by assigning a new suffix; when that happens, update the OELG paths in the comparison configuration to the actual generated map directory. Comparison outputs are protected from replacement unless `--overwrite` is supplied.

The default comparison reads the reproduced paper rasters and project-local validation points. Run `reproduce` first. The comparison reports spatial agreement with the configured points; its validity depends on comparable study areas, input data, and methods. Live OELG instantiation and mapping use their separate configurations; update an alternative comparison configuration explicitly when evaluating new live runs.

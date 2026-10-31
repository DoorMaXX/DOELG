# Instance validation

Run the commands from the KGC_main project root in the KGC environment:

```powershell
conda activate KGC
python main/main_validation.py -t instantiate
python main/main_validation.py -t compare
```

The first command builds an OELG baseline, writes a new snapshot to the configured instance database, exports its audit files, and generates event and factor distance fields. To redraw maps from an existing configured snapshot, run `python main/main_validation.py -t map`. Use `--config PATH` to select an alternative configuration. `--no-plots` skips PNG output; `--output-dir PATH` selects the OELG audit directory for instantiation or the map directory for map-only runs.

Edit `oelg_config.json` for the abstract, observation, and instance databases and snapshot; edit `oelg_distance_config.json` for the OELG map output and `importance_comparison_config.json` for both methods' raster directories and validation points. Map settings shared with KGC_main are read from `../instantiation_config.json`. Existing snapshots and nonempty OELG output folders are preserved by assigning a new suffix; when that happens, update the OELG paths in the comparison configuration to the actual generated map directory. Comparison outputs are protected from replacement unless `--overwrite` is supplied.

The default validation point paths refer to the neighboring KGC_2 project. Change them if the data is stored elsewhere. The comparison reports spatial agreement with the configured points; its validity depends on comparable study areas, input data, and methods.

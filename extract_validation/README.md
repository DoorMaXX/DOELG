# Extraction validation

Run from the KGC_main project root with the KGC environment:

```powershell
python main/main_extraction_validation.py -t evaluate -k event --input data/extraction_validation_dataset/event/default-10shot.xlsx
python main/main_extraction_validation.py -t evaluate -k geo --input data/extraction_validation_dataset/geo/default-10shot.xlsx
python main/main_extraction_validation.py -t evaluate -k relation --input data/extraction_validation_dataset/event_relation/default-10shot.xlsx
python main/main_extraction_validation.py -t plot -k event
python main/main_extraction_validation.py -t table
```

`plot` supports `event`, `geo`, and `relation`; use `--base-dir`, `--output-dir`, and `--shots 1 5 10 30` to select inputs and outputs. By default, inputs are read from `data/extraction_validation_dataset`, and figures are saved under its `results/<kind>` directory. `table` exports all three Table 4 metrics to `results/table4.csv` and does not require `--kind`.

Event evaluation uses the original combined entity/argument micro-F1 from `KGC_2/validation_plot_new.py`. Geographic evaluation uses `KGC_2/validation_geo_all.py` with the `geo/new-2` data version, including relation-word matching and combined entity/relation micro-F1. The evaluation and plotting entry points share these calculators. The source workbooks and SHA-256 hashes are recorded in `data/paper_experiment/provenance.json`.

To prepare a workbook, run `python main/main_extraction_validation.py -t prepare -k event --input INPUT.xlsx --output OUTPUT.xlsx`. The output must be a new file; the input is not modified. `--only-raw` is available for event and relation workbooks. Evaluation expects `raw_json` and `true_json` columns.

During preparation, the existing event and geographic parsers process extraction text. Their behavior, including the existing fallback from missing ground truth to the prediction, is preserved; review missing annotations before interpreting scores.

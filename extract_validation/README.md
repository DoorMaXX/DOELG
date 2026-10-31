# Extraction validation

Run from the KGC_main project root with the KGC environment:

```powershell
python main/main_extraction_validation.py -t evaluate -k event --input data/extraction_validation_dataset/event/default-10shot.xlsx
python main/main_extraction_validation.py -t evaluate -k geo --input data/extraction_validation_dataset/geo/default-10shot.xlsx
python main/main_extraction_validation.py -t evaluate -k relation --input data/extraction_validation_dataset/event_relation/default-10shot.xlsx
python main/main_extraction_validation.py -t plot -k event
```

`plot` supports `event`, `geo`, and `relation`; use `--base-dir`, `--output-dir`, and `--shots 1 5 10 30` to select inputs and outputs. By default, inputs are read from `data/extraction_validation_dataset`, and figures are saved under its `results/<kind>` directory.

To prepare a workbook, run `python main/main_extraction_validation.py -t prepare -k event --input INPUT.xlsx --output OUTPUT.xlsx`. The output must be a new file; the input is not modified. `--only-raw` is available for event and relation workbooks. Evaluation expects `raw_json` and `true_json` columns.

During preparation, the existing event and geographic parsers process extraction text. Their behavior, including the existing fallback from missing ground truth to the prediction, is preserved; review missing annotations before interpreting scores.

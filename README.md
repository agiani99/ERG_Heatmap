# ERG Heatmap Streamlit app

![App_screenshot](https://github.com/agiani99/ERG_Heatmap/blob/main/Screenshot.png)

## What it does

- Upload a CSV (like `activity_CHEMBL202_ERG.csv`).
- The CSV file can be generated from any original file containing these type of columns (ID [e.g. CHEMBLID, CID], SMILES, any activity in form of pchembl or 0/1 class labels). 
- Keeps `pchembl_value` (if present) and only feature columns starting with `fr_` or `ERG...`.
- Drops zero-variance feature columns.
- Creates `class` where `pchembl_value >= 6.5` → 1 else 0.
- Sorts compounds descending by `pchembl_value` (fallback: `class`).
- Renders a normalized (0..1) heatmap for the feature matrix.
- Shows the activity column as a separate strip on the left.

## Run

From this folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```



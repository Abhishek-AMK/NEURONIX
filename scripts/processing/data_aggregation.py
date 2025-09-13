import argparse
import json
import os
import sys
import pandas as pd
import numpy as np

# ---- Custom Aggregation Functions ----
custom_aggregates = {
    "median": np.median,
    "std": np.std,
    "var": np.var,
    "first": lambda x: x.iloc[0] if len(x) > 0 else None,
    "last": lambda x: x.iloc[-1] if len(x) > 0 else None
}

def load_data(file_path):
    """Loads data from CSV, Excel, or JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file type: " + file_path)

def aggregate_data(df, group_by, agg_funcs):
    """
    Aggregates the DataFrame `df` by `group_by` columns using `agg_funcs`.
    agg_funcs: dict of {column: [functions]} or {column: function}
    """
    agg_dict = {}
    for col, funcs in agg_funcs.items():
        if not isinstance(funcs, list):
            funcs = [funcs]
        resolved_funcs = [custom_aggregates.get(f, f) for f in funcs]
        agg_dict[col] = resolved_funcs[0] if len(resolved_funcs) == 1 else resolved_funcs
    result = df.groupby(group_by).agg(agg_dict)
    # Drop duplicate index columns
    for col in result.index.names:
        if col in result.columns:
            result = result.drop(columns=[col])
    result = result.reset_index()
    # Flatten multi-level column names for readability
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = ['_'.join(map(str, col)).strip() for col in result.columns]
    return result

def main():
    parser = argparse.ArgumentParser(description="Data Aggregation Module")
    parser.add_argument("--file", required=True, help="Path to input data file (csv, xlsx, json)")
    parser.add_argument("--group_by", required=True, help="JSON list of columns to group by")
    parser.add_argument("--agg_funcs", required=True, help="JSON dict of {column: [functions]} or {column: function}")
    parser.add_argument("--output", choices=["json", "csv", "excel"], default="json", help="Output format")
    parser.add_argument("--out_path", help="Optional output file path")
    args = parser.parse_args()

    # Parse group_by and agg_funcs
    try:
        group_by = json.loads(args.group_by)
        agg_funcs = json.loads(args.agg_funcs)
    except Exception as e:
        print(f"Error parsing group_by or agg_funcs: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df = load_data(args.file)
        result = aggregate_data(df, group_by, agg_funcs)
    except Exception as e:
        print(f"Data aggregation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.output == "json":
        output = result.to_json(orient="records", indent=2)
        if args.out_path:
            with open(args.out_path, "w", encoding="utf-8") as f:
                f.write(output)
        else:
            print(output)
    elif args.output == "csv":
        if args.out_path:
            result.to_csv(args.out_path, index=False)
        else:
            print(result.to_csv(index=False))
    elif args.output == "excel":
        out_path = args.out_path or "aggregated_result.xlsx"
        result.to_excel(out_path, index=False)
        print(f"Excel file written to {out_path}")

if __name__ == "__main__":
    main()

"""Wrapper around SynGen synthetic data generator."""

import json
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd


SYNGEN_CONFIGS_DIR = Path(__file__).parents[2] / "configs" / "syngen"
RAW_DATA_DIR = Path(__file__).parents[2] / "data" / "raw"


def run_syngen(
    schema_path: str,
    output_path: str,
    rows: Optional[int] = None,
    fmt: str = "csv",
    syngen_cmd: str = "python3 syngen.py",
) -> pd.DataFrame:
    """Run SynGen to generate data from a schema config.

    Args:
        schema_path: Path to SynGen JSON schema file.
        output_path: Where to write the output file.
        rows: Override row count from schema (optional).
        fmt: Output format (csv or json).
        syngen_cmd: Command to invoke SynGen.

    Returns:
        Generated data as a DataFrame.
    """
    with open(schema_path) as f:
        schema = json.load(f)

    if rows is not None:
        schema["rows"] = rows

    # Write temp schema with overrides
    tmp_schema = Path(output_path).parent / "_tmp_schema.json"
    tmp_schema.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_schema, "w") as f:
        json.dump(schema, f)

    cmd = f"{syngen_cmd} --config {tmp_schema} --output {output_path} --format {fmt}"
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: generate locally if SynGen not installed
        df = _generate_fallback(schema)
        if fmt == "csv":
            df.to_csv(output_path, index=False)
        else:
            df.to_json(output_path, orient="records", indent=2)
        return df
    finally:
        tmp_schema.unlink(missing_ok=True)

    if fmt == "csv":
        return pd.read_csv(output_path)
    return pd.read_json(output_path)


def _generate_fallback(schema: dict) -> pd.DataFrame:
    """Standalone fallback generator matching SynGen field types."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = schema.get("rows", 1000)
    data = {}

    for field_def in schema.get("fields", []):
        name = field_def["name"]
        ftype = field_def["type"]
        constraints = field_def.get("constraints", {})

        if ftype == "integer":
            lo = constraints.get("min", 0)
            hi = constraints.get("max", 100)
            data[name] = rng.integers(lo, hi + 1, size=rows)
        elif ftype == "float":
            lo = constraints.get("min", 0.0)
            hi = constraints.get("max", 1.0)
            precision = constraints.get("precision", 2)
            data[name] = np.round(rng.uniform(lo, hi, size=rows), precision)
        elif ftype == "boolean":
            data[name] = rng.choice([True, False], size=rows)
        elif ftype == "string":
            min_len = constraints.get("min_length", 5)
            max_len = constraints.get("max_length", 10)
            data[name] = [
                "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"),
                        size=rng.integers(min_len, max_len + 1)))
                for _ in range(rows)
            ]
        elif ftype == "date":
            start = pd.Timestamp(constraints.get("start", "2025-01-01"))
            end = pd.Timestamp(constraints.get("end", "2050-12-31"))
            days_range = (end - start).days
            data[name] = [
                (start + pd.Timedelta(days=int(rng.integers(0, days_range + 1)))).strftime("%Y-%m-%d")
                for _ in range(rows)
            ]
        else:
            data[name] = rng.integers(0, 100, size=rows)

    return pd.DataFrame(data)


def generate_all_raw_data(rows_override: Optional[int] = None) -> dict[str, pd.DataFrame]:
    """Generate all raw datasets from SynGen configs.

    Returns dict mapping schema name to generated DataFrame.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for schema_file in sorted(SYNGEN_CONFIGS_DIR.glob("*.json")):
        name = schema_file.stem
        output_path = RAW_DATA_DIR / f"{name}.csv"
        df = run_syngen(
            schema_path=str(schema_file),
            output_path=str(output_path),
            rows=rows_override,
        )
        results[name] = df

    return results

from __future__ import annotations

import argparse
import sqlite3
import tarfile
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path("data/chembl-sql-37")
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "chembl_37.db"
DEFAULT_ARCHIVE_PATH = DEFAULT_DATA_DIR / "chembl_37_sqlite.tar.gz"
DEFAULT_TARGETS_OUTPUT = Path("data/chembl_targets.csv")
DEFAULT_BIOACTIVITY_OUTPUT = Path("data/chembl_bioactivities.csv")
REQUIRED_TABLES = {
    "activities",
    "assays",
    "compound_structures",
    "docs",
    "molecule_dictionary",
    "target_components",
    "target_dictionary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Query a local ChEMBL SQLite database for target metadata and \nbioactivity records."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to the extracted ChEMBL SQLite database. Default: {DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=DEFAULT_ARCHIVE_PATH,
        help=(
            "Path to the local ChEMBL SQLite .tar.gz archive. Used only if the "
            "database file is missing."
        ),
    )
    parser.add_argument(
        "--targets-out",
        type=Path,
        default=DEFAULT_TARGETS_OUTPUT,
        help=f"CSV path for target search results. Default: {DEFAULT_TARGETS_OUTPUT}",
    )
    parser.add_argument(
        "--bioactivities-out",
        type=Path,
        default=DEFAULT_BIOACTIVITY_OUTPUT,
        help=(
            "CSV path for bioactivity records for the selected target. "
            f"Default: {DEFAULT_BIOACTIVITY_OUTPUT}"
        ),
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=0,
        help="Which row from the target search results to use. Default: 0",
    )
    parser.add_argument(
        "--standard-type",
        default="IC50",
        help="Bioactivity standard_type filter. Default: IC50",
    )
    parser.add_argument(
        "--standard-relation",
        default="=",
        help="Bioactivity standard_relation filter. Default: =",
    )
    parser.add_argument(
        "--assay-type",
        default="B",
        help="Assay type filter. Default: B",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional LIMIT for the bioactivity query.",
    )

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target-chembl-id",
        help="Exact target ChEMBL ID, for example CHEMBL220.",
    )
    target_group.add_argument(
        "--target-name",
        help="Case-insensitive partial target name search, for example acetylcholinesterase.",
    )
    target_group.add_argument(
        "--uniprot-id",
        help="UniProt accession, for example P22303.",
    )

    return parser.parse_args()


def database_has_required_tables(db_path: Path) -> bool:
    if not db_path.exists() or db_path.stat().st_size == 0:
        return False

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
    except sqlite3.Error:
        return False

    table_names = {row[0] for row in rows}
    return REQUIRED_TABLES.issubset(table_names)


def find_valid_database(search_root: Path) -> Path | None:
    candidates = sorted(
        path for path in search_root.rglob("*.db") if database_has_required_tables(path)
    )
    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    return max(candidates, key=lambda path: path.stat().st_size)


def ensure_database(db_path: Path, archive_path: Path) -> Path:
    if database_has_required_tables(db_path):
        return db_path

    candidate = find_valid_database(db_path.parent)
    if candidate is not None:
        return candidate

    if archive_path.exists():
        print(f"Extracting {archive_path} into {archive_path.parent} ...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=archive_path.parent)

        candidate = find_valid_database(db_path.parent)
        if candidate is not None:
            return candidate

    raw_candidates = sorted(db_path.parent.rglob("*.db"))
    candidate_list = "\n".join(str(path) for path in raw_candidates) or "(none found)"
    raise FileNotFoundError(
        "No valid ChEMBL SQLite database was found after extraction. Checked:\n"
        f"{candidate_list}"
    )


def get_targets(conn: sqlite3.Connection, args: argparse.Namespace) -> pd.DataFrame:
    if args.target_chembl_id:
        sql = """
        SELECT
            td.chembl_id AS target_chembl_id,
            td.pref_name,
            td.organism,
            td.target_type
        FROM target_dictionary td
        WHERE td.chembl_id = ?
        ORDER BY td.chembl_id
        """
        params = (args.target_chembl_id,)
    elif args.target_name:
        sql = """
        SELECT
            td.chembl_id AS target_chembl_id,
            td.pref_name,
            td.organism,
            td.target_type
        FROM target_dictionary td
        WHERE lower(td.pref_name) LIKE lower(?)
        ORDER BY td.pref_name, td.chembl_id
        """
        params = (f"%{args.target_name}%",)
    else:
        sql = """
        SELECT DISTINCT
            td.chembl_id AS target_chembl_id,
            td.pref_name,
            td.organism,
            td.target_type
        FROM target_dictionary td
        JOIN target_components tc
            ON td.tid = tc.tid
        JOIN component_sequences cs
            ON tc.component_id = cs.component_id
        WHERE cs.accession = ?
        ORDER BY td.pref_name, td.chembl_id
        """
        params = (args.uniprot_id,)

    return pd.read_sql_query(sql, conn, params=params)


def get_bioactivities(
    conn: sqlite3.Connection,
    target_chembl_id: str,
    standard_type: str,
    standard_relation: str,
    assay_type: str,
    limit: int | None,
) -> pd.DataFrame:
    sql = """
    SELECT
        act.activity_id,
        ass.chembl_id AS assay_chembl_id,
        ass.description AS assay_description,
        ass.assay_type,
        md.chembl_id AS molecule_chembl_id,
        cs.canonical_smiles,
        act.standard_type,
        act.standard_relation,
        act.standard_value,
        act.standard_units,
        act.pchembl_value,
        td.chembl_id AS target_chembl_id,
        td.pref_name AS target_pref_name,
        td.organism AS target_organism,
        d.chembl_id AS document_chembl_id,
        d.journal AS document_journal,
        d.year AS document_year
    FROM activities act
    JOIN assays ass
        ON act.assay_id = ass.assay_id
    JOIN target_dictionary td
        ON ass.tid = td.tid
    JOIN molecule_dictionary md
        ON act.molregno = md.molregno
    LEFT JOIN compound_structures cs
        ON md.molregno = cs.molregno
    LEFT JOIN docs d
        ON ass.doc_id = d.doc_id
    WHERE td.chembl_id = ?
      AND act.standard_type = ?
      AND act.standard_relation = ?
      AND ass.assay_type = ?
    ORDER BY act.activity_id
    """

    params: list[str | int] = [
        target_chembl_id,
        standard_type,
        standard_relation,
        assay_type,
    ]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    return pd.read_sql_query(sql, conn, params=params)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fetch_chembl_data(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    archive_path: Path = DEFAULT_ARCHIVE_PATH,
    target_chembl_id: str | None = None,
    target_name: str | None = None,
    uniprot_id: str | None = None,
    target_index: int = 0,
    standard_type: str = "IC50",
    standard_relation: str = "=",
    assay_type: str = "B",
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    search_values = [target_chembl_id, target_name, uniprot_id]
    if sum(value is not None for value in search_values) != 1:
        raise ValueError(
            "Provide exactly one of target_chembl_id, target_name, or uniprot_id."
        )

    query_args = argparse.Namespace(
        target_chembl_id=target_chembl_id,
        target_name=target_name,
        uniprot_id=uniprot_id,
    )

    resolved_db_path = ensure_database(db_path, archive_path)
    with sqlite3.connect(resolved_db_path) as conn:
        targets = get_targets(conn, query_args)
        if targets.empty:
            raise ValueError("No targets matched the supplied search criteria.")

        if target_index < 0 or target_index >= len(targets):
            raise IndexError(
                f"target_index {target_index} is out of range for {len(targets)} targets."
            )

        selected_target = targets.iloc[target_index]
        bioactivities = get_bioactivities(
            conn=conn,
            target_chembl_id=selected_target["target_chembl_id"],
            standard_type=standard_type,
            standard_relation=standard_relation,
            assay_type=assay_type,
            limit=limit,
        )

    return targets, bioactivities


def write_csv_outputs(
    targets: pd.DataFrame,
    bioactivities: pd.DataFrame,
    targets_out: Path,
    bioactivities_out: Path,
) -> None:
    ensure_parent_dir(targets_out)
    targets.to_csv(targets_out, index=False)

    ensure_parent_dir(bioactivities_out)
    bioactivities.to_csv(bioactivities_out, index=False)


def main() -> None:
    args = parse_args()
    db_path = ensure_database(args.db_path, args.archive_path)
    print(f"Using database: {db_path}")

    try:
        targets, bioactivities = fetch_chembl_data(
            db_path=db_path,
            archive_path=args.archive_path,
            target_chembl_id=args.target_chembl_id,
            target_name=args.target_name,
            uniprot_id=args.uniprot_id,
            target_index=args.target_index,
            standard_type=args.standard_type,
            standard_relation=args.standard_relation,
            assay_type=args.assay_type,
            limit=args.limit,
        )
    except (ValueError, IndexError) as exc:
        raise SystemExit(str(exc)) from exc

    write_csv_outputs(
        targets=targets,
        bioactivities=bioactivities,
        targets_out=args.targets_out,
        bioactivities_out=args.bioactivities_out,
    )

    print(f"Wrote {len(targets)} target rows to {args.targets_out}")
    print(targets.head(min(len(targets), 10)).to_string(index=False))
    selected_target = targets.iloc[args.target_index]
    print(
        "Selected target: "
        f"{selected_target['target_chembl_id']} | {selected_target['pref_name']} | "
        f"{selected_target['organism']}"
    )
    print(
        f"Wrote {len(bioactivities)} bioactivity rows to {args.bioactivities_out}"
    )
    if not bioactivities.empty:
        print(bioactivities.head(min(len(bioactivities), 10)).to_string(index=False))


if __name__ == "__main__":
    main()

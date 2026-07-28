"""
chembl_local_client.py
 
Local ChEMBL database client: downloads the official ChEMBL SQLite
distribution once, then lets you query it with a fluent, chembl_webresource_client-
style API (`.filter(**kwargs)`) instead of hitting the network on every call.
All query methods return pandas DataFrames.
 
Example
-------
>>> client = ChEMBLLocalClient(data_dir="./chembl_data")
>>> client.download_database()          # no-op if already downloaded
>>> df = (
...     client.activities
...     .filter(target_chembl_id="CHEMBL203", standard_type="IC50")
...     .filter(pchembl_value__isnull=False)
...     .to_df()
... )
>>> egfr_targets = client.search_target("EGFR")
"""

from __future__ import annotations
 
import re
import shutil
import sqlite3
import tarfile
from pathlib import Path
from typing import Any, Iterable, Optional
 
import pandas as pd
import requests


def extract_downloaded_tarball(
    tarball_path: str | Path,
    destination_dir: str | Path,
    *,
    remove_archive: bool = False,
) -> Path:
    """
    Extract an already-downloaded .tar.gz archive into destination_dir.

    Returns the destination directory so callers can chain path discovery after
    extraction.
    """
    tarball = Path(tarball_path)
    destination = Path(destination_dir)

    if not tarball.exists():
        raise FileNotFoundError(f"Tarball not found: {tarball}")

    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(destination)

    if remove_archive:
        tarball.unlink(missing_ok=True)

    return destination


# --------------------------------------------------------------------------- #
# Fluent query builder (mimics chembl_webresource_client's .filter() style)
# --------------------------------------------------------------------------- #

_LOOKUP_TO_SQL = {
    "exact": "=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "icontains": "LIKE",
    "contains": "LIKE",
    "startswith": "LIKE",
    "in": "IN",
    "isnull": None,  # handled specially
}

class QuerySet:
    """
    Lazy, chainable query against a single table in the locally-saved ChEMBL DB.
    Mirrors Django-ORM-like lookup syntax used by the chembl_webresource_client,
    e.g. filter(pchembl_value__gte=6, standard_type="IC50").
    """

    
    def __init__(self, conn: sqlite3.Connection, table: str, columns: Optional[Iterable[str]] = None):
        self._conn = conn
        self._table = table
        self._columns = list(columns) if columns else ["*"]
        self._where: list[str] = []
        self._params: list[Any] = []
        self._order_by: Optional[str] = None
        self._limit: Optional[int] = None
 
    # -- building queries ------------------------------------------------- #

    def filter(self, **kwargs) -> QuerySet:
        """Add WHERE conditions. 
        Supports: 
            field, 
            field__gt, 
            field__gte, 
            field__lt,
            field__lte, 
            field__in, 
            field__icontains, 
            field__contains, 
            field__startswith,
            field__isnull 
            Lookups are chainable like the ChEMBL web resource client."""
        
        for key, value in kwargs.items():
            if "__" in key:
                field, lookup = key.rsplit("__", 1)
            else:
                field, lookup = key, "exact"
 
            if lookup == "isnull":
                self._where.append(f'"{field}" IS {"NULL" if value else "NOT NULL"}')
                continue
 
            if lookup == "in":
                placeholders = ",".join("?" for _ in value)
                self._where.append(f'"{field}" IN ({placeholders})')
                self._params.extend(value)
                continue
 
            op = _LOOKUP_TO_SQL.get(lookup)
            if op is None:
                raise ValueError(f"Unsupported lookup: {lookup}")
 
            if lookup == "icontains":
                self._where.append(f'"{field}" LIKE ? COLLATE NOCASE')
                self._params.append(f"%{value}%")
            elif lookup == "contains":
                self._where.append(f'"{field}" LIKE ?')
                self._params.append(f"%{value}%")
            elif lookup == "startswith":
                self._where.append(f'"{field}" LIKE ?')
                self._params.append(f"{value}%")
            else:
                self._where.append(f'"{field}" {op} ?')
                self._params.append(value)
        return self
    
    def only(self, *columns: str) -> "QuerySet":
        """Restrict the returned columns."""
        self._columns = list(columns)
        return self
    
    def order_by(self, column: str, descending: bool = False) -> "QuerySet":
        self._order_by = f'"{column}" {"DESC" if descending else "ASC"}'
        return self
 
    def limit(self, n: int) -> "QuerySet":
        self._limit = n
        return self
    

    # -- execution --------------------------------------------------------- #
 
    def _sql(self) -> str:
        cols = ", ".join(self._columns)
        sql = f'SELECT {cols} FROM "{self._table}"'
        if self._where:
            sql += " WHERE " + " AND ".join(self._where)
        if self._order_by:
            sql += f" ORDER BY {self._order_by}"
        if self._limit is not None:
            sql += f" LIMIT {self._limit}"
        return sql
 
    def to_df(self) -> pd.DataFrame:
        """Execute the query and return the results as a DataFrame."""
        return pd.read_sql_query(self._sql(), self._conn, params=self._params)
 
    def count(self) -> int:
        cols_backup = self._columns
        self._columns = ["COUNT(*) as n"]
        try:
            return int(self.to_df().iloc[0]["n"])
        finally:
            self._columns = cols_backup

    def __repr__(self) -> str:
        return f"<QuerySet table={self._table!r} sql={self._sql()!r}>"



# --------------------------------------------------------------------------- #
# Main client
# --------------------------------------------------------------------------- #
 
class ChEMBLLocalClient:
    """
    Downloads and queries a local copy of the ChEMBL SQLite database.
 
    Query methods are designed to feel like the chembl_webresource_client API
    (`client.activities.filter(...).to_df()`) but run entirely against the
    local SQLite file, so there's no rate limiting and no network latency
    once the database is downloaded.
    """
 
    RELEASES_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases"
    LATEST_INDEX_URL = f"{RELEASES_BASE_URL}/../latest/"
    
    def __init__(self, data_dir: str = "./chembl_data", version: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.version = version
        self._conn: Optional[sqlite3.Connection] = None
 
    # ------------------------------------------------------------------ #
    # Download / setup
    # ------------------------------------------------------------------ #
    def _db_candidates(self) -> list[Path]:
        """Any .db files already present under data_dir (any ChEMBL version)."""
        return sorted(self.data_dir.rglob("chembl_*.db"))
 
    def is_downloaded(self) -> bool:
        return len(self._db_candidates()) > 0
 
    @property
    def db_path(self) -> Path:
        candidates = self._db_candidates()
        if not candidates:
            raise FileNotFoundError(
                "No local ChEMBL database found. Call download_database() first."
            )
        # prefer the highest version number if multiple are present
        def version_key(p: Path) -> int:
            m = re.search(r"chembl_(\d+)", p.name)
            return int(m.group(1)) if m else -1
 
        return sorted(candidates, key=version_key)[-1]
    
    def _resolve_download_url(self) -> str:
        """Work out the tarball URL for self.version, or auto-detect the latest release."""
        if self.version:
            v = self.version
            return f"{self.RELEASES_BASE_URL}/chembl_{v}/chembl_{v}_sqlite.tar.gz"
 
        resp = requests.get(self.LATEST_INDEX_URL, timeout=30)
        resp.raise_for_status()
        versions = re.findall(r"chembl_(\d+)_sqlite\.tar\.gz", resp.text)
        if not versions:
            raise RuntimeError(
                "Could not auto-detect the latest ChEMBL version from "
                f"{self.LATEST_INDEX_URL}. Pass an explicit version=, e.g. "
                'ChEMBLLocalClient(version="35").'
            )
        self.version = max(versions, key=int)
        return f"{self.LATEST_INDEX_URL}chembl_{self.version}_sqlite.tar.gz"
 
    def download_database(self, force: bool = False, chunk_size: int = 1 << 20) -> Path:
        """
        Download and extract the ChEMBL SQLite database if it isn't already present.
        Returns the path to the local .db file. Safe to call repeatedly -- it is a
        no-op (unless force=True) once the database has been downloaded.
        """
        if self.is_downloaded() and not force:
            print(f"ChEMBL database already present at {self.db_path}")
            return self.db_path
 
        url = self._resolve_download_url()
        tarball_path = self.data_dir / Path(url).name
        print(f"Downloading {url} ...")
 
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            written = 0
            with open(tarball_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = 100 * written / total
                        print(f"\r  {written / 1e9:.2f} GB / {total / 1e9:.2f} GB ({pct:.1f}%)", end="")
        print("\nDownload complete. Extracting...")
 
        extract_downloaded_tarball(
            tarball_path,
            self.data_dir,
            remove_archive=True,
        )

        if not self.is_downloaded():
            raise RuntimeError(
                "Extraction finished but no chembl_*.db file was found under "
                f"{self.data_dir}. Check the archive contents manually."
            )
        print(f"ChEMBL database ready at {self.db_path}")
        return self.db_path
 

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #
 
    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self.is_downloaded():
                raise FileNotFoundError(
                    "Database not downloaded yet. Call download_database() first."
                )
            self._conn = sqlite3.connect(str(self.db_path))
        return self._conn
 
    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
 
    def __enter__(self) -> "ChEMBLLocalClient":
        self.connect()
        return self
 
    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # ChEMBL-API-style entry points -> QuerySet
    # ------------------------------------------------------------------ #

    # activities, molecules, compound_structures, targets, assays, documents
    @property
    def activities(self) -> QuerySet:
        return QuerySet(self.connect(), "activities")
 
    @property
    def molecules(self) -> QuerySet:
        return QuerySet(self.connect(), "molecule_dictionary")
 
    @property
    def compound_structures(self) -> QuerySet:
        return QuerySet(self.connect(), "compound_structures")
 
    @property
    def targets(self) -> QuerySet:
        return QuerySet(self.connect(), "target_dictionary")
 
    @property
    def assays(self) -> QuerySet:
        return QuerySet(self.connect(), "assays")
 
    @property
    def documents(self) -> QuerySet:
        return QuerySet(self.connect(), "docs")
 
    # ------------------------------------------------------------------ #
    # Raw SQL escape hatch
    # ------------------------------------------------------------------ #
    
    def query(self, sql: str, params: Optional[Iterable[Any]] = None) -> pd.DataFrame:
        """Run arbitrary SQL and get a DataFrame back."""
        return pd.read_sql_query(sql, self.connect(), params=params or [])
 
    # ------------------------------------------------------------------ #
    # Convenience methods (common ChEMBL API-style lookups)
    # ------------------------------------------------------------------ #
 
    def search_target(self, name: str, organism: Optional[str] = None) -> pd.DataFrame:
        """Search targets by (partial) preferred name, similar to
        `new_client.target.filter(pref_name__icontains=name)`."""
        qs = self.targets.filter(pref_name__icontains=name)
        if organism:
            qs = qs.filter(organism__icontains=organism)
        return qs.to_df()
 
    def search_molecule(self, name: str) -> pd.DataFrame:
        """Search molecules by (partial) preferred name."""
        return self.molecules.filter(pref_name__icontains=name).to_df()
 
    def get_activities_for_target(
        self,
        target_chembl_id: str,
        standard_types: Optional[Iterable[str]] = None,
        pchembl_value_min: Optional[float] = None,
        include_structures: bool = True,
    ) -> pd.DataFrame:
        """
        Convenience method returning activities for a target joined with molecule
        preferred names and (optionally) canonical SMILES -- the shape you'd get
        back from `new_client.activity.filter(target_chembl_id=..., standard_type__in=...)`
        but pulled from the local database and pre-joined.
        """
        joins = """
            FROM activities a
            JOIN assays s ON a.assay_id = s.assay_id
            JOIN molecule_dictionary m ON a.molregno = m.molregno
        """
        cols = [
            "a.activity_id",
            "a.molregno",
            "m.chembl_id AS molecule_chembl_id",
            "m.pref_name AS molecule_pref_name",
            "s.chembl_id AS assay_chembl_id",
            "s.assay_type",
            "a.standard_type",
            "a.standard_relation",
            "a.standard_value",
            "a.standard_units",
            "a.pchembl_value",
            "s.target_chembl_id" if False else "a.target_chembl_id" if False else None,
        ]
        # target_chembl_id lives on assays in ChEMBL's schema
        cols = [c for c in cols if c]
        cols.append("s.target_chembl_id")
 
        if include_structures:
            joins += " JOIN compound_structures cs ON a.molregno = cs.molregno"
            cols.append("cs.canonical_smiles")
 
        where = ['s.target_chembl_id = ?']
        params: list[Any] = [target_chembl_id]
 
        if standard_types:
            placeholders = ",".join("?" for _ in standard_types)
            where.append(f"a.standard_type IN ({placeholders})")
            params.extend(standard_types)
 
        if pchembl_value_min is not None:
            where.append("a.pchembl_value >= ?")
            params.append(pchembl_value_min)
 
        sql = f"SELECT {', '.join(cols)} {joins} WHERE {' AND '.join(where)}"
        return self.query(sql, params)
 
    def get_molecule(self, chembl_id: str) -> pd.DataFrame:
        """Look up a single molecule by its ChEMBL ID, including structure."""
        sql = """
            SELECT m.*, cs.canonical_smiles, cs.standard_inchi, cs.standard_inchi_key
            FROM molecule_dictionary m
            LEFT JOIN compound_structures cs ON m.molregno = cs.molregno
            WHERE m.chembl_id = ?
        """
        return self.query(sql, [chembl_id])
 
    def table_names(self) -> list[str]:
        """List every table in the local database (useful for exploring the schema)."""
        df = self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return df["name"].tolist()
 
 
if __name__ == "__main__":

    extract_downloaded_tarball(
        tarball_path="./chembl_data/chembl_37_sqlite.tar.gz",
        destination_dir="./chembl_data",
        remove_archive=False,
    )

    # # Minimal smoke test / usage example.
    # client = ChEMBLLocalClient(data_dir="./chembl_data")
    # client.download_database()  # no-op if already downloaded
 
    # with client:
    #     egfr = client.search_target("Epidermal growth factor receptor", organism="Homo sapiens")
    #     print(egfr[["chembl_id", "pref_name", "organism"]].head())
 
    #     if not egfr.empty:
    #         target_id = egfr.iloc[0]["chembl_id"]
    #         acts = client.get_activities_for_target(
    #             target_id, standard_types=["IC50"], pchembl_value_min=5.0
    #         )
    #         print(acts.head())
 

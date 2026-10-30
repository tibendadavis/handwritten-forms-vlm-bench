"""Read-only DHIS2 client: data set metadata and org unit lookup. It never writes to DHIS2."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from formbench.files import slugify, write_json

DATASET_FIELDS = ",".join(
    [
        "id",
        "name",
        "periodType",
        "categoryCombo[id,name,categoryOptionCombos[id,name]]",
        "sections[id,name,sortOrder,dataElements[id]]",
        "dataSetElements["
        "categoryCombo[id,name,categoryOptionCombos[id,name]],"
        "dataElement[id,name,formName,valueType,"
        "categoryCombo[id,name,categoryOptionCombos[id,name]]]]",
    ]
)


class Dhis2Client:
    def __init__(self, base_url: str, username: str, password: str, timeout: float = 30.0) -> None:
        self._http = httpx.Client(
            base_url=f"{base_url.rstrip('/')}/api",
            auth=(username, password),
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    @classmethod
    def from_env(cls) -> Dhis2Client:
        try:
            return cls(
                os.environ["DHIS2_BASE_URL"],
                os.environ["DHIS2_USERNAME"],
                os.environ["DHIS2_PASSWORD"],
            )
        except KeyError as missing:
            raise SystemExit(f"Missing {missing.args[0]} in .env") from None

    def get_dataset(self, uid: str) -> dict[str, Any]:
        response = self._http.get(f"/dataSets/{uid}", params={"fields": DATASET_FIELDS})
        response.raise_for_status()
        return response.json()

    def find_org_units_by_code(self, code: str) -> list[dict[str, Any]]:
        response = self._http.get(
            "/organisationUnits",
            params={"filter": f"code:eq:{code}", "fields": "id,name,code", "paging": "false"},
        )
        response.raise_for_status()
        return response.json()["organisationUnits"]


def fetch_metadata(dataset: str, out: Path | None = None) -> Path:
    """Save a data set's metadata (elements, category option combos, attribute options) as
    schemas/<data set name>/metadata.json, the offline source of truth for counting."""
    ds = Dhis2Client.from_env().get_dataset(dataset)
    path = (out or Path("schemas") / slugify(ds["name"])) / "metadata.json"
    write_json(path, ds)
    return path


def main() -> None:
    import argparse

    from dotenv import find_dotenv, load_dotenv

    parser = argparse.ArgumentParser(description="Download DHIS2 data set metadata (read-only).")
    parser.add_argument("--dataset", required=True, help="DHIS2 data set UID")
    parser.add_argument("--out", type=Path, help="folder (default schemas/<data set name>)")
    args = parser.parse_args()
    load_dotenv(find_dotenv(usecwd=True))
    path = fetch_metadata(args.dataset, args.out)
    ds = json.loads(path.read_text())
    print(f"{ds['name']}: {len(ds['dataSetElements'])} data elements -> {path}")

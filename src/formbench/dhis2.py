from __future__ import annotations

import os
from typing import Any

import httpx

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

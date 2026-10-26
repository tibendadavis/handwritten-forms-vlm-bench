import argparse
import json
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv

from formbench.dhis2 import Dhis2Client
from formbench.schema_gen import build_extraction_schema, build_fields, build_mapping, slugify


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate extraction schema + DHIS2 mapping.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="DHIS2 data set UID to fetch")
    source.add_argument("--snapshot", type=Path, help="regenerate offline from a metadata.json")
    parser.add_argument("--out", type=Path, default=Path("schemas"))
    args = parser.parse_args()

    if args.snapshot:
        ds = json.loads(args.snapshot.read_text())
    else:
        load_dotenv(find_dotenv(usecwd=True))
        ds = Dhis2Client.from_env().get_dataset(args.dataset)

    fields = build_fields(ds)
    out = args.out / slugify(ds["name"])
    out.mkdir(parents=True, exist_ok=True)

    (out / "metadata.json").write_text(json.dumps(ds, indent=2, ensure_ascii=False) + "\n")
    schema = build_extraction_schema(ds, fields)
    (out / "extract.schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n"
    )
    mapping = build_mapping(ds, fields)
    (out / "mapping.yaml").write_text(yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True))

    sections = len({f.section for f in fields if f.section})
    print(f"{ds['name']}: {len(fields)} fields across {sections} sections -> {out}/")


if __name__ == "__main__":
    main()

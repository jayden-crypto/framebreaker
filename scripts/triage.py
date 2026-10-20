"""Print every new finding compactly: data, pipeline (as Polars code), distinct outcomes."""
import json
import sys
from pathlib import Path

from framebreaker.render import pipeline_code

root = Path(sys.argv[1])
for line in (root / "index.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r.get("known"):
        continue
    d = root / r["dir"]
    case = json.loads((d / "case.json").read_text())
    print("=" * 90)
    print(r["id"], r["kind"], r["tags"])
    for i, t in enumerate(case["tables"]):
        print(f"  t{i}", t["columns"], t["rows"][:6], "..." if len(t["rows"]) > 6 else "")
    if case.get("tlp"):
        print("  tlp pred:", json.dumps(case["tlp"]))
    print("  " + pipeline_code(case["pipeline"], "pl").replace("\n", "\n  "))
    readme = (d / "README.md").read_text()
    print("  " + readme.split("## Outcomes", 1)[-1].strip().replace("\n", "\n  ")[:900] if "## Outcomes" in readme else readme.split("## Error",1)[-1][:300])

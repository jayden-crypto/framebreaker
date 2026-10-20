"""Ad-hoc exploration: fuzz for a while, shrink every failure, group by fingerprint."""
import collections
import json
import random
import sys

from framebreaker.compare import show_rows
from framebreaker.engines import make_backends
from framebreaker.gen import Generator
from framebreaker.ir import data_tags, features
from framebreaker.oracle import check, fingerprint, tlp_valid
from framebreaker.shrink import shrink

seed, n = int(sys.argv[1]), int(sys.argv[2])
bs = make_backends()
rng = random.Random(seed)
g = Generator(rng)
groups = collections.defaultdict(list)
for _ in range(n):
    case = g.gen_case()
    v = check(case, bs)
    if not v:
        continue
    fp = fingerprint(v)
    def fails(c, fp=fp):
        w = check(c, bs)
        return w is not None and fingerprint(w) == fp
    small, evals = shrink(case, fails, tlp_valid, budget=600)
    w = check(small, bs)
    key = (fp, tuple(features(small)))
    groups[key].append((small, w))
for (fp, feats), items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    small, w = items[0]
    print("=" * 100)
    print(len(items), fp[0], [g for g in fp[1]], feats, data_tags(small))
    print(json.dumps(small["tables"]), json.dumps(small["pipeline"]))
    seen = []
    for name, o in w["outcomes"].items():
        txt = show_rows(o, 6)
        if txt not in seen:
            seen.append(txt)
            print(f"  [{name}]", txt.replace("\n", " // "))

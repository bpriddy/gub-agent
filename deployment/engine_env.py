"""Print a deployed Agent Engine's baked-in env as `KEY=value` pairs.

Reads the reasoningEngines GET response on stdin. Prints a `<none: ...>` marker
rather than nothing when `deploymentSpec` carries no env, so the caller's log
shows what the resource actually looks like.
"""

import json
import sys

engine = json.load(sys.stdin)
spec = engine.get("spec", {}).get("deploymentSpec", {})
pairs = [f"{v['name']}={v.get('value', '')}" for v in (spec.get("env") or [])]
print(" ".join(pairs) if pairs else "<none: deploymentSpec=" + json.dumps(spec) + ">")

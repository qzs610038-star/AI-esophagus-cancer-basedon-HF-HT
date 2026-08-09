"""无训练模板入口，用于验证 W### 包可独立加载。"""

from __future__ import annotations

import json
from pathlib import Path


def load_config(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    config_path = Path(__file__).parents[1] / "configs" / "smoke.json"
    print(json.dumps(load_config(config_path), ensure_ascii=False, sort_keys=True))

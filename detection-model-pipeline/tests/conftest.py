"""将仓库根的 code/ 与项目根加入 sys.path，供测试导入本地模块。"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT / "code"), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.local_image_edit import request_local_image_edit


def handle_request(message: dict[str, Any]) -> dict[str, Any]:
    request_id = str(message.get("request_id") or "")
    try:
        config = message.get("config")
        if not isinstance(config, dict):
            raise ValueError("worker request config must be a JSON object")
        response = request_local_image_edit(
            str(message.get("prompt") or ""),
            str(message.get("image_url") or ""),
            config,
        )
        return {"request_id": request_id, "ok": True, "response": response}
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        return {"request_id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("worker request must be a JSON object")
            response = handle_request(message)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            response = {"request_id": "", "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()

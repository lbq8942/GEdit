import json
import os
from datetime import datetime

import requests

PUSHPLUS_URL = "https://www.pushplus.plus/send"

def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    return getter(key, default) if getter is not None else getattr(cfg, key, default)

def send_pushplus(cfg, title, content):
    """Send an optional PushPlus notification without interrupting the task."""
    if not bool(_cfg_get(cfg, "pushplus_enabled", False)):
        return False
    token = str(_cfg_get(cfg, "pushplus_token", "") or "").strip()
    if not token:
        token_env = str(_cfg_get(cfg, "pushplus_token_env", "PUSHPLUS_TOKEN"))
        token = os.environ.get(token_env, "").strip()
    if not token:
        print("[PushPlus] skipped: no token configured")
        return False
    payload = {
        "token": token,
        "title": str(title)[:100],
        "content": str(content),
        "template": "markdown",
        "channel": str(_cfg_get(cfg, "pushplus_channel", "wechat")),
    }
    option = _cfg_get(cfg, "pushplus_option", None)
    if option:
        payload["option"] = str(option)
    try:
        response = requests.post(
            PUSHPLUS_URL, json=payload,
            timeout=float(_cfg_get(cfg, "pushplus_timeout", 10)),
        )
        response.raise_for_status()
        result = response.json()
        if result.get("code") != 200:
            raise RuntimeError(result.get("msg") or result)
        print("[PushPlus] notification sent")
        return True
    except Exception as exc:
        print(f"[PushPlus] warning: notification failed: {exc}")
        return False

def notify_finished(cfg, elapsed_seconds, metrics=None, result_file=None):
    content = [
        "- Status: finished",
        f"- Runtime: {elapsed_seconds / 3600:.3f} hours",
        f"- Model: {_cfg_get(_cfg_get(cfg, 'llms', None), 'name', 'unknown')}",
        f"- Algorithm: {_cfg_get(_cfg_get(cfg, 'algs', None), 'name', 'unknown')}",
        f"- Dataset: {_cfg_get(cfg, 'data', 'unknown')}",
        f"- Number of edits: {_cfg_get(cfg, 'num_edits', 'unknown')}",
    ]
    if result_file:
        content.append(f"- Result file: `{result_file}`")
    if metrics is not None:
        content.extend([
            "", "## Metrics", "```json",
            json.dumps(metrics, ensure_ascii=False, indent=2), "```",
        ])
    return send_pushplus(cfg, "GEdit finished", "\n".join(content))

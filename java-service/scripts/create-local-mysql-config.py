"""Create ignored local development credentials and one-time MySQL setup SQL."""

from __future__ import annotations

import json
from pathlib import Path
import secrets


local = Path(__file__).resolve().parents[1] / ".local"
local.mkdir(exist_ok=True)
credentials_path = local / "mysql.json"
if credentials_path.exists():
    raise SystemExit(f"Credentials already exist: {credentials_path}")

credentials = {
    "rootPassword": secrets.token_urlsafe(24),
    "appUser": "rag",
    "appPassword": secrets.token_urlsafe(24),
    "port": 3307,
}
credentials_path.write_text(json.dumps(credentials, indent=2), encoding="utf-8")

sql = f"""\
CREATE DATABASE IF NOT EXISTS modular_rag CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
CREATE USER IF NOT EXISTS 'rag'@'localhost' IDENTIFIED BY '{credentials['appPassword']}';
CREATE USER IF NOT EXISTS 'rag'@'127.0.0.1' IDENTIFIED BY '{credentials['appPassword']}';
GRANT ALL PRIVILEGES ON modular_rag.* TO 'rag'@'localhost';
GRANT ALL PRIVILEGES ON modular_rag.* TO 'rag'@'127.0.0.1';
ALTER USER 'root'@'localhost' IDENTIFIED BY '{credentials['rootPassword']}';
"""
(local / "mysql-setup.sql").write_text(sql, encoding="utf-8")
(local / "mysql-client.cnf").write_text(
    "[client]\n"
    "host=127.0.0.1\n"
    "port=3307\n"
    "user=rag\n"
    f"password={credentials['appPassword']}\n",
    encoding="utf-8",
)
print("Created local MySQL credentials and setup SQL under .local")

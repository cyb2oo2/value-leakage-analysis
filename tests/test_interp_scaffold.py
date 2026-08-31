from __future__ import annotations

import json
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INTERP = ROOT / "interp"


class InterpScaffoldTests(unittest.TestCase):
    def test_default_project_has_no_dependencies(self) -> None:
        with (INTERP / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)
        self.assertEqual(project["project"]["dependencies"], [])
        self.assertFalse(project["tool"]["uv"]["package"])

    def test_snapshot_keeps_all_expensive_actions_disabled(self) -> None:
        snapshot = json.loads(
            (INTERP / "compatibility_snapshot.json").read_text(encoding="utf-8")
        )
        self.assertTrue(snapshot["upstream"]["commit"])
        self.assertTrue(snapshot["verification_state"]["upstream_source_reviewed"])
        self.assertTrue(all(value is False for value in snapshot["guards"].values()))

    def test_checker_emits_json_without_heavy_imports(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(INTERP / "check_environment.py"), "--json"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=20,
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(
            report["safety"],
            {
                "api_keys_read": False,
                "downloads": False,
                "heavy_package_imports": False,
                "network_calls": False,
            },
        )


if __name__ == "__main__":
    unittest.main()

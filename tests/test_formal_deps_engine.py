import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "formal_deps_engine.py"
spec = importlib.util.spec_from_file_location("formal_deps_engine", MODULE_PATH)
formal_deps_engine = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = formal_deps_engine
spec.loader.exec_module(formal_deps_engine)

DepEngine = formal_deps_engine.DepEngine
ToolState = formal_deps_engine.ToolState


class FormalDepsEngineInstallFlowTests(unittest.TestCase):
    def _missing_states(self):
        return {
            "tlc": ToolState(False, None, None),
            "coqc": ToolState(False, None, None),
        }

    def test_install_detects_403_and_emits_local_artifact_guidance(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            manifest_path = tmp / "deps_manifest.json"
            install_script = tmp / "install.sh"
            install_script.write_text("#!/usr/bin/env bash\nexit 1\n")
            install_script.chmod(0o755)

            engine = DepEngine(manifest_path=manifest_path, install_script=install_script)
            engine.preflight = self._missing_states  # type: ignore[assignment]
            engine.run_install_script = lambda: (
                False,
                "",
                "curl: (22) The requested URL returned error: 403",
            )  # type: ignore[assignment]

            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, {}, clear=False))
                stack.enter_context(mock.patch.dict(os.environ, {"LOCAL_ARTIFACT_ROOT": ""}, clear=False))
                out = io.StringIO()
                err = io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    rc = engine.install()

            self.assertEqual(rc, 1)
            stdout = out.getvalue()
            self.assertIn("Detected HTTP 403 during network install.", stdout)
            self.assertIn("export LOCAL_ARTIFACT_ROOT=/path/to/pre-mirrored-artifacts", stdout)
            self.assertIn("LOCAL_ARTIFACT_ROOT is not set", stdout)

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["history"][-2]["stage"], "network_install")
            self.assertFalse(manifest["history"][-2]["success"])
            self.assertEqual(manifest["history"][-1]["stage"], "local_fallback")
            self.assertFalse(manifest["history"][-1]["success"])
            self.assertEqual(manifest["history"][-1]["details"]["http_status"], 403)

    def test_second_install_skips_network_after_failed_manifest_state(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            manifest_path = tmp / "deps_manifest.json"
            install_script = tmp / "install.sh"
            install_script.write_text("#!/usr/bin/env bash\nexit 0\n")
            install_script.chmod(0o755)

            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "history": [
                            {
                                "time_utc": "2026-01-01T00:00:00+00:00",
                                "stage": "network_install",
                                "success": False,
                                "details": {"stderr_tail": "HTTP 403"},
                            }
                        ],
                    }
                )
            )

            engine = DepEngine(manifest_path=manifest_path, install_script=install_script)
            engine.preflight = self._missing_states  # type: ignore[assignment]

            def _should_not_run_network():
                raise AssertionError("network install should have been skipped")

            engine.run_install_script = _should_not_run_network  # type: ignore[assignment]

            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, {"LOCAL_ARTIFACT_ROOT": ""}, clear=False))
                out = io.StringIO()
                err = io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    rc = engine.install()

            self.assertEqual(rc, 1)
            stdout = out.getvalue()
            self.assertIn(
                "Skipping network install based on previous failed network attempt and no LOCAL_ARTIFACT_ROOT.",
                stdout,
            )
            self.assertIn("LOCAL_ARTIFACT_ROOT is not set", stdout)

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["history"][-1]["stage"], "local_fallback")
            self.assertFalse(manifest["history"][-1]["success"])


if __name__ == "__main__":
    unittest.main()

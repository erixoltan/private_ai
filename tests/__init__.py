"""Tests never read or write the user's own config, keys or approval rules."""

import atexit
import os
import shutil
import tempfile

_config = tempfile.mkdtemp(prefix="private_ai_test_config_")
atexit.register(shutil.rmtree, _config, ignore_errors=True)
os.environ["PRIVATE_AI_CONFIG"] = os.path.join(_config, "config.toml")
os.environ["PRIVATE_AI_BROWSER"] = "off"  # and never start the real browser; tests that want one supply their own

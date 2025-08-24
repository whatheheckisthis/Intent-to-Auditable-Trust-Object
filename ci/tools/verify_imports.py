# ci/import_check.py

try:
    import numpy
    import flake8
    import pytest
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

print("✅ All critical packages imported successfully.")

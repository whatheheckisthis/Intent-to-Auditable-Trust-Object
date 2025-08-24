def test_imports():
    for pkg in ["numpy", "pytest", "flake8"]:
        __import__(pkg)

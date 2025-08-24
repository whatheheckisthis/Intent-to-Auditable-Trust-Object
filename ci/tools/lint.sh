steps:
  - uses: actions/checkout@v4

  - name: Set up Miniconda
    uses: conda-incubator/setup-miniconda@v3
    with:
      miniforge-variant: Miniforge3
      miniforge-version: 23.3.1-0  # (Optional but recommended to pin)
      python-version: 3.10
      activate-environment: testenv
      environment-file: environment.yml
      auto-activate-base: false

  - name: Install flake8 and pytest
    shell: bash -l {0}
    run: |
      conda install flake8 pytest

  - name: Lint codebase
    shell: bash -l {0}
    run: |
      bash ci/lint.sh

  - name: CI environment check
    shell: bash -l {0}
    run: |
      bash ci/run_all_checks.sh

  - name: Test with pytest
    shell: bash -l {0}
    run: |
      pytest

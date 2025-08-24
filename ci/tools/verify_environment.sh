#!/bin/bash
# CI environment verification script


- name: Verify CI environment setup and imports
  shell: bash -l {0}
  run: |
    bash ci/verify_environment.sh

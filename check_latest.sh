#!/bin/bash
CITE_COLLECTION=${CITE_COLLECTION:-caselaw_f2d} \
  cat $(ls -t mlruns/0/*/artifacts/output.txt | head -1) | python cite_check.py

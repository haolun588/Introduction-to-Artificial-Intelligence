#!/usr/bin/env bash
# Wrapper: 5-fold CV with sqrt-inverse class weights (outputs/cv5_sqrt, outputs/cv5_sqrt_oof).
exec bash tools/run_cv5.sh cv5_sqrt sqrt-inverse "$@"

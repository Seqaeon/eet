#!/bin/bash
export LD_LIBRARY_PATH=/home/seqaeon/Downloads/nanochat/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
/home/seqaeon/Downloads/nanochat/.venv/bin/python -m scripts.research_compare --depth 2 --run-dir /tmp/test_dir

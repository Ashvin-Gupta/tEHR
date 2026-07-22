#!/usr/bin/env python3
"""
Backward-compatible entry point for ``subgroup_analysis``.

Accepts legacy flags (--results_json / --model_name) as well as the new
--primary_json / --compare_json interface.
"""
from __future__ import annotations

import argparse
import sys

from src.analysis import subgroup_analysis


def main():
    # Legacy CLI → new CLI
    if "--results_json" in sys.argv and "--primary_json" not in sys.argv:
        argv = []
        skip = False
        for i, arg in enumerate(sys.argv):
            if skip:
                skip = False
                continue
            if arg == "--results_json":
                argv.extend(["--primary_json", sys.argv[i + 1]])
                skip = True
            elif arg == "--model_name":
                argv.extend(["--primary_name", sys.argv[i + 1]])
                skip = True
            elif arg == "--compare_results_json":
                argv.extend(["--compare_json", sys.argv[i + 1]])
                skip = True
            else:
                argv.append(arg)
        sys.argv = argv

    p = argparse.ArgumentParser(description="Subgroup analysis (legacy wrapper)")
    p.add_argument("--config_filepath", required=True)
    p.add_argument("--primary_json", required=True)
    p.add_argument("--primary_name", default="text_llm")
    p.add_argument("--compare_json", default=None)
    p.add_argument("--compare_name", default="encoder")
    p.add_argument("--comparison_pair", default="custom")
    p.add_argument("--split", default="held_out")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--save_per_example", action="store_true")
    args = p.parse_args()
    subgroup_analysis.run(args)


if __name__ == "__main__":
    main()

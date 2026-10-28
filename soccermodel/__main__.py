"""
Command line:

    python -m soccermodel update japan            # full matchday run
    python -m soccermodel update japan --sims 2000  # quicker run while checking
    python -m soccermodel check japan             # validate data only (team names, scores)
    python -m soccermodel shadow greece           # after update: price with the other engine, compare
"""

import argparse
import sys
from pathlib import Path

from .data import DataError, load_config, load_initial_elo, load_results, validate
from .pipeline import ROOT, update
from .shadow import shadow


def main(argv=None):
    parser = argparse.ArgumentParser(prog='soccermodel')
    parser.add_argument('command', choices=['update', 'check', 'shadow'])
    parser.add_argument('country', help='folder name under data/, e.g. japan')
    parser.add_argument('--sims', type=int, default=None, help='override number of simulations')
    parser.add_argument('--engine', default='hierarchical', help='shadow: rating engine to compare (default hierarchical)')
    args = parser.parse_args(argv)

    country_dir = ROOT / 'data' / args.country
    try:
        if args.command == 'check':
            cfg = load_config(country_dir)
            results = load_results(cfg)
            validate(results, load_initial_elo(cfg), cfg)
            print(f"OK: {len(results)} results, latest {results['date'].max():%Y-%m-%d}")
        elif args.command == 'shadow':
            shadow(country_dir, ROOT / 'output' / args.country, engine=args.engine, n_sims=args.sims)
        else:
            update(country_dir, ROOT / 'output' / args.country, n_sims=args.sims)
    except DataError as e:
        print(f"DATA ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

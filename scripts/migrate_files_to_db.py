"""
One-shot, idempotent migration of legacy file records into MySQL (standalone CLI).

The migration also runs automatically once on app startup (see db/startup.py).

Usage: python scripts/migrate_files_to_db.py [--logs-dir logs] [--dry-run]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import test_connection  # noqa: E402
from db.migration import Report, run_migration  # noqa: E402
from db.schema import init_schema  # noqa: E402


def main() -> int:
    """CLI entry point: connect, ensure schema, migrate, print the report."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--dry-run", action="store_true", help="roll back instead of committing")
    args = ap.parse_args()
    if not test_connection():
        return 1
    init_schema()
    result = run_migration(args.logs_dir, args.dry_run)
    if args.dry_run:
        print("(dry run — nothing committed)")
    rep = Report()
    rep.rows = result["by_source"]
    rep.print()
    for err in result["errors"]:
        print(f"ERROR: {err}")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())

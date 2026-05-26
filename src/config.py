from pathlib import Path

# Project root is the parent of this file's directory (src/)
DB_PATH = Path(__file__).resolve().parent.parent / "netrics.db"

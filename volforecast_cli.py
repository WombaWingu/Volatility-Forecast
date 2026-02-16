"""
Standalone CLI entrypoint for volatility forecasting.
Run: python volforecast_cli.py daily --ticker NVDA
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root and src to path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

# Now import the CLI module
from volforecast.cli import main

if __name__ == "__main__":
    main()

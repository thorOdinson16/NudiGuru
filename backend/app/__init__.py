"""NudiGuru backend application package."""
import sys

# Windows consoles default to a legacy codepage that cannot encode Kannada.
# Reconfigure the standard streams so logging/printing non-ASCII text cannot
# crash the process, regardless of the host codepage.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - stream not configurable
            pass

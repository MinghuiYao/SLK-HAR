"""Backward-compatible SLK-HAR model entry point."""

from slk_har.legacy import SLK_Net
from slk_har.models import SLKNet

__all__ = ["SLKNet", "SLK_Net"]

"""itksnap-mcp: agent-callable interface to ITK-SNAP human-in-the-loop segmentation.

Model proposes, human disposes. See README.md.
"""

from .dls_client import DLSClient, AutomaticResult, load_nifti_for_upload

__all__ = ["DLSClient", "AutomaticResult", "load_nifti_for_upload"]
__version__ = "0.0.1"

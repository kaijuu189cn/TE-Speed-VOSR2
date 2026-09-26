"""TE-Speed-VOSR2 — VOSR2 超分辨率放大（跨平台）

Windows: 使用编译的 .pyd（原始 C++ 实现）
Linux:   使用纯 Python nodes_linux.py（等效算法）
"""

import logging

logger = logging.getLogger("TE-Speed-VOSR2")
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = None

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    logger.info("Loaded compiled nodes.pyd (Windows native)")
except Exception:
    try:
        from .nodes_linux import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        logger.info("Loaded pure Python nodes_linux.py (Linux compatible)")
    except Exception as e:
        logger.error(f"Failed to load any node implementation: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "1.0.1"

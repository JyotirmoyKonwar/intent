"""M3: find what a skill's code, config and hidden text can do (PLAN.md §6.3).

Everything here reads text and ASTs only. Nothing is imported or executed.
"""

from skillalign.behaviors.detect import Detection, detect_behaviors

__all__ = ["Detection", "detect_behaviors"]

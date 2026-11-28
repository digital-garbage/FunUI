"""Schedule. A node-only module: the list of noise levels a run walks down, picked by name."""

from .nodes import FunUISchedule

ID = "schedule"
TITLE = "Schedule"
STAGE = "sampling"
CATEGORY = "system"
STATUS = "experimental"

NODES = [FunUISchedule]

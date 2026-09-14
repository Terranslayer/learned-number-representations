# symemerge/varg/__init__.py
"""VAR-GROUNDING BRANCH (spec docs/design.md).

Branch-local package. It imports `symemerge/world/*` READ-ONLY and imports NOTHING from
`symemerge/agent/*` (spec 1: zero parameter sharing with the main line; spec 9:
TEST-BENCH PERP DUT). Nothing here is main-line era machinery.

DISCLOSURE OF RECORD (spec 1): this branch has NO eye-brain-hand body plan -- no hand, no
note, no canvas medium, no motor noise, no write->read state wipe. It is a perception +
generation model only, and the number is GIVEN as a continuous scalar. No result from this
branch is quotable as "the model learned to count".
"""

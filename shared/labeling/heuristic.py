# shim — canonical definitions live in workers (worker-specific code, not shared)
#   workers.core.labeling.heuristic  (assign_structural/semantic, apply_two_layer_labels)
#   workers.infrastructure.audio.features  (build_segment_descriptors*)
from workers.core.labeling.heuristic import *  # noqa: F401,F403
from workers.core.labeling.heuristic import (
    apply_two_layer_labels,
    assign_semantic_labels,
    assign_structural_labels,
)
from workers.infrastructure.audio.features import (
    build_segment_descriptors,
    build_segment_descriptors_from_audio,
)

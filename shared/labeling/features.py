# shim — canonical definition lives in workers.core.labeling.features
from workers.core.labeling.features import *  # noqa: F401,F403
from workers.core.labeling.features import (  # explicit re-export
    build_segment_label_vectors,
    feature_names,
)

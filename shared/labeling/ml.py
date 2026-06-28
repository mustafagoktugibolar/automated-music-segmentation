# shim — canonical definition lives in workers.core.labeling.ml
from workers.core.labeling.ml import *  # noqa: F401,F403
from workers.core.labeling.ml import predict_semantic_labels, reset_model_cache  # explicit re-export

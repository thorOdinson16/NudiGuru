import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
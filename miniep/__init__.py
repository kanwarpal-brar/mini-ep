MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"

# Correctness-gate constants shared by the reference generator and the gates.
GATE_LOGIT_TOL = 2e-3  # max abs logit diff tolerated vs HF (fp32 summation-order noise)
LOGIT_FP_SIZE = 64  # width of the final-position logit fingerprint
LOGIT_FP_KEY = "last_logits_head64"  # reference.json field (name must track LOGIT_FP_SIZE)

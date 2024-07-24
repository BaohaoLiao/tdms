from models.standard_model import StandardModel

MODEL_FACTORY = {
    "densenet201": StandardModel,
    "tf_efficientnet_b3.ns_jft_in1k": StandardModel,
}
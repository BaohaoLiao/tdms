from models.standard_model import StandardModel

MODEL_FACTORY = {
    "densenet201": StandardModel,
    "tf_efficientnet_b3_ns": StandardModel,
    "tf_efficientnetv2_b3.in21k_ft_in1k": StandardModel,
    "resnet34": StandardModel,
}
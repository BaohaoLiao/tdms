from models.standard_model import StandardModel

MODEL_FACTORY = {
    "densenet169": StandardModel,
    "densenet201": StandardModel,
    "densenet161": StandardModel,
    "tf_efficientnet_b3.ns_jft_in1k": StandardModel,
    "tf_efficientnet_b4.ns_jft_in1k": StandardModel,
    "tf_efficientnet_b5.ns_jft_in1k": StandardModel,
    "tf_efficientnetv2_b3.in21k_ft_in1k": StandardModel,
    "resnet34": StandardModel,
    "inception_v3.tf_in1k": StandardModel,
    "xception": StandardModel,
    "coat_lite_small": StandardModel,
    "coat_small": StandardModel,
}
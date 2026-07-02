from transformers import PretrainedConfig


class RampNetConfig(PretrainedConfig):
    model_type = "rampnet"

    def __init__(
        self,
        input_size=(2048, 4096),
        heatmap_size=(512, 1024),
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
        recommended_threshold=0.55,
        recommended_min_distance=10,
        tta_recommended=True,
        **kwargs,
    ):
        self.input_size = list(input_size)
        self.heatmap_size = list(heatmap_size)
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)
        self.recommended_threshold = recommended_threshold
        self.recommended_min_distance = recommended_min_distance
        self.tta_recommended = tta_recommended
        super().__init__(**kwargs)

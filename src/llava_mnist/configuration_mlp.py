from transformers import PretrainedConfig


class MLPConfig(PretrainedConfig):
    model_type = "mlp"

    def __init__(
        self,
        input_size: int = 784,
        output_size: int = 4096,
        **kwargs,
    ):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__(**kwargs)


MLPConfig.register_for_auto_class()

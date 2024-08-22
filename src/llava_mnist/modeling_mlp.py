import torch

from transformers import PreTrainedModel

from configuration_mlp import MLPConfig


class MLP(PreTrainedModel):
    config_class = MLPConfig

    def __init__(self, config):
        super().__init__(config)
        self.input_layer = torch.nn.Linear(config.input_size, config.output_size)

    def forward(self, inputs):
        x = self.input_layer(inputs)
        return x


MLP.register_for_auto_class("AutoModel")

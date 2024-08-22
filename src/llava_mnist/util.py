import torch
from transformers import AutoModelForCausalLM, AutoModel
from transformers import AutoTokenizer
import re


def build_multi_modal_prompt(
    prompt: str,
    image: torch.Tensor,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    vision_model: AutoModel,
) -> torch.Tensor:
    """
    Insert the image into the <image> token in the prompt and return the multi-modal embedding
    """
    parts = prompt.split("<image>")
    prefix = tokenizer(parts[0])
    suffix = tokenizer(parts[1])
    # embed the prefix and suffix
    prefix_embedding = model.get_input_embeddings()(torch.tensor(prefix["input_ids"]))
    suffix_embedding = model.get_input_embeddings()(torch.tensor(suffix["input_ids"]))
    # embed the image
    image_embedding = vision_model(image).to(torch.bfloat16).to(model.device)
    # concatenate the embeddings
    multi_modal_embedding = torch.cat(
        [prefix_embedding, image_embedding, suffix_embedding], dim=0
    )

    return multi_modal_embedding


DIGIT_IN_ENGLISH = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def extract_first_number(s: str) -> int | None:
    match = re.search(r"\d+", s)
    if match:
        return int(match.group())
    else:
        return None

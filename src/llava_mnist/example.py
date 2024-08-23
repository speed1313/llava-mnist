from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from torchvision import transforms
import util
from transformers import AutoModel


def build_multi_modal_prompt(
    prompt: str,
    image: torch.Tensor,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    vision_model: AutoModel,
) -> torch.Tensor:
    parts = prompt.split("<image>")
    prefix = tokenizer(parts[0])
    suffix = tokenizer(parts[1])
    prefix_embedding = model.get_input_embeddings()(torch.tensor(prefix["input_ids"]))
    suffix_embedding = model.get_input_embeddings()(torch.tensor(suffix["input_ids"]))
    image_embedding = vision_model(image).to(torch.bfloat16).to(model.device)
    multi_modal_embedding = torch.cat(
        [prefix_embedding, image_embedding, suffix_embedding], dim=0
    )
    return multi_modal_embedding


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

vision_model = AutoModel.from_pretrained("speed/llava-mnist", trust_remote_code=True)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|>"
user_prompt = "<|start_header_id|>user<|end_header_id|>"
question = "<image>What digit is this?"
assistant_prompt = "<|start_header_id|>assistant<|end_header_id|>"

prompt = system_prompt + user_prompt + question + assistant_prompt

ds = load_dataset("ylecun/mnist", split="test")


def transform_image(examples):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )
    examples["pixel_values"] = [transform(image) for image in examples["image"]]

    return examples


ds.set_transform(transform=transform_image)


model.eval()
vision_model.eval()

example = ds[0]

input_embeded = util.build_multi_modal_prompt(
    prompt, example["pixel_values"].unsqueeze(0), tokenizer, model, vision_model
).unsqueeze(0)
response = model.generate(
    inputs_embeds=input_embeded,
    max_new_tokens=20,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = response[0]
print("Label:", example["label"])
answer = tokenizer.decode(response, skip_special_tokens=True)
print("Answer:", answer)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
import torch
from configuration_mlp import MLPConfig
from modeling_mlp import MLP
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import util

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]


system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|>"
user_prompt = "<|start_header_id|>user<|end_header_id|>"
question = "<image>What digit is this?"
assistant_prompt = "<|start_header_id|>assistant<|end_header_id|>"

prompt = system_prompt + user_prompt + question + assistant_prompt


ds = load_dataset("ylecun/mnist", split="train")
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ]
)


def transforms(examples):
    examples["pixel_values"] = [transform(image) for image in examples["image"]]

    return examples


ds = ds.select(range(1000))

ds.set_transform(transforms)

vision_model = MLP(MLPConfig())

# training loop
vision_model.train()
model.eval()

optimizer = torch.optim.AdamW(vision_model.parameters(), lr=1e-3)


for i in tqdm(range(1000)):
    optimizer.zero_grad()
    answer_text = f"The digit is {ds[i]['label']}.<|eot_id|>"
    answer = tokenizer(answer_text, return_tensors="pt")
    true_answer_input_ids = answer["input_ids"]
    input_embeded = util.build_multi_modal_prompt(
        prompt, ds[i]["pixel_values"].unsqueeze(0), tokenizer, model, vision_model
    ).unsqueeze(0)
    output_embeded = model.get_input_embeddings()(answer["input_ids"])
    input_embeded = torch.cat([input_embeded, output_embeded], dim=1)
    labels = torch.zeros(input_embeded.shape[1], dtype=torch.long)
    labels[:] = -100
    # TODO: mask only <image> token
    answer_length = answer["input_ids"].shape[1]
    labels[-answer_length:] = true_answer_input_ids[0]
    outputs = model(inputs_embeds=input_embeded, labels=labels)
    loss = outputs.loss

    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(loss.item())
        print("true answer:", answer_text)
        print(
            tokenizer.decode(
                torch.argmax(outputs.logits, dim=-1)[0], skip_special_tokens=True
            )
        )


vision_model.push_to_hub("llava-mnist", private=True)
# TODO: check llm's parameters do not change


# TODO: check the image's embedding and digit's embedding

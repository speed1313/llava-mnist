from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from configuration_mlp import MLPConfig
from modeling_mlp import MLP
from torchvision import transforms
from tqdm import tqdm
import util
import wandb


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


train_num = 1000
ds = ds.select(range(train_num))

ds.set_transform(transforms)

mlp_config = MLPConfig()
vision_model = MLP(mlp_config)

# training loop
vision_model.train()
model.eval()

optimizer = torch.optim.AdamW(vision_model.parameters(), lr=1e-3)

wandb.init(project="llava-mnist")


for i, example in tqdm(enumerate(ds)):
    optimizer.zero_grad()
    answer_text = f"The digit is {example['label']}.<|eot_id|>"
    answer = tokenizer(answer_text, return_tensors="pt")
    true_answer_input_ids = answer["input_ids"]
    input_embeded = util.build_multi_modal_prompt(
        prompt, example["pixel_values"].unsqueeze(0), tokenizer, model, vision_model
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
        # generate response
        response = model.generate(
            inputs_embeds=input_embeded[:, :-answer_length],
            max_new_tokens=20,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response_text = tokenizer.decode(response[0], skip_special_tokens=True)
        print("response:", response_text)
        # wandb HTML logging
        wandb.log(
            {
                "iter": i,
                "train/loss": loss.item(),
                "Generated Text": wandb.Html(
                    "<p>Ground truth: "
                    + answer_text
                    + "</p><p>Response: "
                    + response_text
                    + "</p>"
                ),
            }
        )


vision_model.push_to_hub("llava-mnist", private=True)
mlp_config.push_to_hub("llava-mnist", private=True)

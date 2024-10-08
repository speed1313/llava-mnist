from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm
import util
from transformers import AutoModel
import click


@click.command()
@click.option(
    "--question", default="What digit is this?", help="Question to ask the model"
)
def main(question: str):
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    system_prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|>"
    )
    user_prompt = "<|start_header_id|>user<|end_header_id|>"
    question = f"<image>{question}"
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

    test_num = 1000
    ds = ds.select(range(test_num))

    ds.set_transform(transform=transform_image)

    vision_model = AutoModel.from_pretrained(
        "speed/llava-mnist", trust_remote_code=True
    )
    # evaluation
    model.eval()
    vision_model.eval()

    accuracy = 0
    for i, example in enumerate(tqdm(ds)):
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
        print()
        prediction = util.extract_first_number(answer)
        if (
            prediction == example["label"]
            or util.DIGIT_IN_ENGLISH[example["label"]] in answer
        ):
            accuracy += 1
        # cosine similarity between vision_embed and digit's embedding
        digit_emb = torch.mean(
            model.get_input_embeddings()(
                torch.tensor(tokenizer.convert_tokens_to_ids(str(example["label"])))
            ),
            dim=0,
        )
        vision_embed = (
            vision_model(example["pixel_values"].unsqueeze(0))
            .to(torch.bfloat16)
            .to(model.device)
        )
        print(torch.nn.CosineSimilarity()(vision_embed, digit_emb.unsqueeze(0)))
        if i % 100 == 0:
            print(f"Accuracy: {accuracy/(i+1)}")
    print(f"Accuracy: {accuracy/test_num}")


if __name__ == "__main__":
    main()

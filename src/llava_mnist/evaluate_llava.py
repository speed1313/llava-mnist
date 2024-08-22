from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset
from torchvision import transforms
import torch
from tqdm import tqdm
import util

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

prompt = "USER: <image>\nWhat digit is this? ASSISTANT:"


ds = load_dataset("ylecun/mnist", split="test")


ds = ds.select(range(1000))


accuracy = 0


for i, example in enumerate(tqdm(ds)):
    image = example["image"]

    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    generate_ids = model.generate(**inputs, max_new_tokens=15)
    output = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print("Prediction:", output)
    print("Label:", example["label"])
    # cosine similarity between vision_embed and digit's embedding
    # digit_emb = torch.mean(model.get_input_embeddings()(torch.tensor(processor.convert_tokens_to_ids(str(example["label"]))), device="cuda"), dim=0)
    # vision_embed = model.vision_model(image.unsqueeze(0)).to(torch.bfloat16).to("cuda")
    # print("Cosine similarity:", torch.nn.CosineSimilarity()(vision_embed, digit_emb.unsqueeze(0)))
    # extract the digit from the output
    prediction = util.extract_first_number(output)
    # digit in english table

    if (
        prediction == example["label"]
        or util.DIGIT_IN_ENGLISH[example["label"]] in output
    ):
        accuracy += 1
    if i % 100 == 0:
        print("Accuracy:", accuracy / (i + 1))

print("Final accuracy:", accuracy / len(ds))

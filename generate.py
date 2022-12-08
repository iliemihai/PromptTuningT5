from transformers import AutoTokenizer
from model import T5PromptTuningLM

name = "dumitrescustefan/mt5-large-romanian"
tokenizer = AutoTokenizer.from_pretrained(name)
model = T5PromptTuningLM.from_pretrained(name, soft_prompt_path="./soft_prompt/soft_prompt.model")
model.eval()

text = ""Filmul a fost ingrozitor de bun. Nu am decat cuvinte de lauda despre el."""

input_ids = tokenizer.encode(text, return_tensors="pt")

outputs = model.generate(input_ids=input_ids)
output_text = tokenizer.decode(outputs[0])
print(output_text)

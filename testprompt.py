from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "openai-community/gpt2"  # You can try "gpt2-medium", "gpt2-large", etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_completion(prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=input_length + max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):]  # Only return the completion part

# Interactive loop
print("Enter a prompt (or type 'exit' to quit):")
while True:
    prompt = input(">> ")
    if prompt.lower() in {"exit", "quit"}:
        break
    output = generate_completion(prompt)
    print("\n--- Completion ---")
    print(output.strip())
    print("------------------\n")


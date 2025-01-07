from unsloth import FastLanguageModel

# Set important parameters
max_seq_length = 2048  # Choose the maximum sequence length
dtype = None  # Automatically detect data type. Use Float16 for Tesla T4, V100, or Bfloat16 for Ampere+ GPUs
load_in_4bit = True  # Use 4bit quantization to reduce memory usage

# Load the pre-trained model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="gabapppp/naruto-chatbot-model",  # The model you have trained
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Enable native 2x faster inference
FastLanguageModel.for_inference(model)

# Define a sample prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Tokenize the input
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "What is a famous tall tower in Paris?",  # Instruction
            "",  # Input
            "",  # Output - leave empty for generation!
        )
    ], return_tensors="pt").to("cuda")  # Move inputs to GPU (cuda)

# Generate output from the model
outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)

# Decode and display the output
decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(decoded_output)

# If you need to save the model and tokenizer
if True:
    # Save the model and tokenizer in GGUF format
    model.save_pretrained_gguf("model", tokenizer)
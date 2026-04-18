# ============================================================
# TEST SCRIPT FOR YOUR LORA FINE-TUNED SmolLM2-360M
# ============================================================
# What is this file?
#   After you've trained your model using train_lora.py,
#   run this script to chat with it and see how it responds!
#
# How to run:
#   python test_lora.py
#
# Make sure the OUTPUT_DIR below matches what you set in train_lora.py
# ============================================================


import torch  # PyTorch — the core deep learning library

# Hugging Face Transformers — for loading the model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# PEFT — the library that handles loading LoRA adapter weights
from peft import PeftModel


# ============================================================
# CONFIGURATION — Make sure these match your train_lora.py settings
# ============================================================

# This is the original base model (same one used in training)
BASE_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

# This is the folder where your trained LoRA weights were saved
LORA_MODEL_DIR = "./lora_output"

# --- Generation Settings ---
# These control HOW the model generates text responses.

# max_new_tokens: Maximum number of new words/tokens the model will generate.
# If the model seems to cut off too early, increase this number.
MAX_NEW_TOKENS = 200

# temperature: Controls randomness/creativity of responses.
# - 0.1 = very focused and repetitive (deterministic)
# - 0.7 = balanced creativity (recommended)
# - 1.5 = very random and creative (may be incoherent)
TEMPERATURE = 0.7

# top_p: "Nucleus sampling" — the model only picks from the top tokens
# whose combined probability adds up to top_p.
# 0.9 means it only considers the most likely 90% of options at each step.
TOP_P = 0.9

# do_sample: If True, the model randomly samples from possible next tokens
# (more creative). If False, it always picks the most likely token (more robotic).
DO_SAMPLE = True

# repetition_penalty: Penalizes the model for repeating itself.
# 1.0 = no penalty, 1.2 = mild penalty (recommended), 2.0 = strong penalty
REPETITION_PENALTY = 1.2


# ============================================================
# STEP 1: LOAD THE BASE MODEL AND TOKENIZER
# ============================================================

print("=" * 60)
print("Loading model... (this may take a moment)")
print("=" * 60)

# Load the tokenizer from the saved LoRA folder.
# It should be identical to the original, but saved locally for convenience.
tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_DIR)

# Make sure we have a padding token (same fix as in training)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the original base model (the unmodified SmolLM2-360M)
print(f"Loading base model: {BASE_MODEL_NAME}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float32  # Use 32-bit floats for CPU
)

# ============================================================
# STEP 2: LOAD AND APPLY YOUR LORA ADAPTER WEIGHTS
# ============================================================

# PeftModel.from_pretrained loads your trained LoRA adapter
# and merges it ON TOP of the base model.
# Think of it like installing a plugin on top of the original software.
print(f"Loading LoRA adapter from: {LORA_MODEL_DIR}")
model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR)

# Set the model to "evaluation mode" — this turns off training-specific
# behaviors like dropout, so responses are consistent.
model.eval()

print("\n✅ Model loaded and ready!")
print("=" * 60)


# ============================================================
# STEP 3: DEFINE THE RESPONSE GENERATION FUNCTION
# ============================================================

def generate_response(user_input: str) -> str:
    """
    Takes a user's question/message and returns the model's response.

    Parameters:
        user_input (str): The text message from the user.

    Returns:
        str: The model's generated response text.
    """

    # Format the input the same way we did during training.
    # This is important! The model learned from text in this format,
    # so we need to use the same format at test time.
    prompt = f"User: {user_input}\nAssistant:"

    # Tokenize the prompt: convert the text into a list of token IDs (numbers).
    # return_tensors="pt" means return PyTorch tensors (the format the model needs).
    inputs = tokenizer(
        prompt,
        return_tensors="pt",   # Return as PyTorch tensor
        truncation=True,        # Cut off if the prompt is too long
        max_length=256          # Maximum input length (same as training)
    )

    # torch.no_grad() tells PyTorch we're NOT training right now.
    # This saves memory and speeds things up during inference (generation).
    with torch.no_grad():
        # model.generate() is the function that actually produces the response.
        output_ids = model.generate(
            input_ids=inputs["input_ids"],          # The tokenized prompt
            attention_mask=inputs["attention_mask"], # Tells model which tokens to pay attention to
            max_new_tokens=MAX_NEW_TOKENS,           # How many tokens to generate
            temperature=TEMPERATURE,                 # Creativity level
            top_p=TOP_P,                             # Nucleus sampling threshold
            do_sample=DO_SAMPLE,                     # Use random sampling
            repetition_penalty=REPETITION_PENALTY,  # Avoid repetition
            pad_token_id=tokenizer.pad_token_id,    # Token used for padding
            eos_token_id=tokenizer.eos_token_id,    # Stop generating at this token
        )

    # The output_ids include BOTH the input tokens AND the newly generated tokens.
    # We only want the NEW tokens (the model's response), so we slice off the input part.
    # input_ids.shape[1] is the length of the input, so we take everything after it.
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]

    # Decode the token IDs back into human-readable text.
    # skip_special_tokens=True removes tokens like <eos> from the output.
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Clean up any leading/trailing whitespace
    return response_text.strip()


# ============================================================
# STEP 4: INTERACTIVE CHAT LOOP
# ============================================================

print("\n🤖 Your fine-tuned AI is ready to chat!")
print("   Type your message and press Enter to get a response.")
print("   Type 'quit' or 'exit' to stop.")
print("   Type 'reset' to clear conversation history.")
print("=" * 60)
print()

# This is an infinite loop that keeps the conversation going
# until the user types 'quit' or 'exit'.
while True:
    try:
        # Get input from the user
        # The "\n" before "You:" adds a blank line for readability
        user_message = input("\nYou: ").strip()

        # Skip empty inputs (if user just presses Enter)
        if not user_message:
            print("(Please type something!)")
            continue

        # Exit commands — stop the loop
        if user_message.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye! 👋")
            break

        # Reset command — just a friendly message (no conversation history in this simple version)
        if user_message.lower() == "reset":
            print("(Conversation reset — note: this model doesn't retain memory between messages anyway)")
            continue

        # Generate a response from the model
        print("\nAssistant: ", end="", flush=True)  # Print without newline so response appears right after
        response = generate_response(user_message)
        print(response)

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\n\nInterrupted. Goodbye! 👋")
        break


# ============================================================
# QUICK BATCH TEST (runs before the chat loop if uncommented)
# ============================================================
# If you want to test with a fixed set of questions instead of
# typing interactively, uncomment the block below and comment out
# the while loop above.

# test_questions = [
#     "What is your name?",
#     "How can I help you today?",
#     "Tell me something interesting.",
# ]
#
# print("\n--- BATCH TEST MODE ---")
# for question in test_questions:
#     print(f"\nUser: {question}")
#     answer = generate_response(question)
#     print(f"Assistant: {answer}")
#     print("-" * 40)

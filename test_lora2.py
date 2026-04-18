# ============================================================
# QUICK TEST SCRIPT — SINGLE COMMAND LINE INPUT
# ============================================================
# How to use:
#   python test_lora2.py "Your question here"
#
# Example:
#   python test_lora2.py "What is an apple?"
#
# Unlike test_lora.py which opens an interactive chat loop,
# this script takes your question directly from the command line,
# prints one response, and then exits immediately.
# ============================================================


import sys    # sys.argv lets us read arguments passed in the command line
import torch  # PyTorch — needed to load and run the model

from transformers import AutoTokenizer, AutoModelForCausalLM  # Load model/tokenizer
from peft import PeftModel                                     # Load LoRA adapter


# ============================================================
# STEP 1: READ THE QUESTION FROM THE COMMAND LINE
# ============================================================

# sys.argv is a list of everything typed in the command line.
# sys.argv[0] = the script name (e.g. "test_lora2.py")
# sys.argv[1] = the first argument (your question)
#
# Example: python test_lora2.py "What is an apple?"
#   sys.argv[0] = "test_lora2.py"
#   sys.argv[1] = "What is an apple?"

if len(sys.argv) < 2:
    # len(sys.argv) < 2 means no argument was provided after the script name
    print("Usage: python test_lora2.py \"Your question here\"")
    print("Example: python test_lora2.py \"What is an apple?\"")
    sys.exit(1)  # Exit with error code 1 (means something went wrong)

# Join all extra words in case the user forgot quotes and typed multiple words
# e.g. python test_lora2.py What is an apple?
# would give: sys.argv[1:] = ["What", "is", "an", "apple?"]
# " ".join() stitches them back into one string: "What is an apple?"
user_input = " ".join(sys.argv[1:])


# ============================================================
# CONFIGURATION — Must match your train_lora.py settings
# ============================================================

BASE_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"  # The original base model
LORA_MODEL_DIR  = "./lora_output"                 # Folder with your trained adapter

# Generation settings
MAX_NEW_TOKENS     = 200   # Maximum number of tokens (words) to generate
TEMPERATURE        = 0.2   # Creativity: 0.1 = focused, 1.5 = very random
TOP_P              = 0.9   # Only sample from the top 90% likely next tokens
DO_SAMPLE          = True  # Use random sampling (more natural responses)
REPETITION_PENALTY = 1.2   # Penalize repeated words/phrases


# ============================================================
# STEP 2: LOAD THE MODEL (same as test_lora.py)
# ============================================================

print("Loading model...")

# Load the tokenizer from your saved LoRA output folder
tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_DIR)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float32
)

# Load and apply your trained LoRA adapter on top of the base model
model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR)

# Set to evaluation mode — disables dropout so output is consistent
model.eval()

print("Model ready.\n")


# ============================================================
# STEP 3: GENERATE A RESPONSE
# ============================================================

# Format the input the same way as during training
prompt = f"User: {user_input}\nAssistant:"

# Tokenize the prompt into numbers the model can read
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=256
)

# Generate the response — torch.no_grad() saves memory during inference
with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=DO_SAMPLE,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# Slice off the input tokens — we only want the newly generated response
new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]

# Decode the token IDs back into readable text
response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================
# STEP 4: PRINT THE RESULT
# ============================================================

print(f"You: {user_input}")
print(f"Assistant: {response}")

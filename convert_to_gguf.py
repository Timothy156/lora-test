# ============================================================
# CONVERT TRAINED LORA MODEL TO GGUF FORMAT
# ============================================================
# What is this file?
#   After training with train_lora.py, this script:
#     1. Loads your base model + LoRA adapter
#     2. MERGES them into a single, standalone model
#     3. Converts that merged model into GGUF format
#        (so you can run it in llama.cpp, Ollama, LM Studio, etc.)
#
# What is GGUF?
#   GGUF is a file format designed for running AI models efficiently
#   on consumer hardware (CPU, low VRAM). It's used by tools like
#   llama.cpp, Ollama, and LM Studio.
#
# What is "merging"?
#   During LoRA training, your adapter weights are kept SEPARATE
#   from the base model. Before converting, we need to bake/merge
#   them back together into ONE model file that llama.cpp can read.
#
# Requirements:
#   pip install transformers peft torch
#   You also need llama.cpp cloned/built on your machine.
#   Get it here: https://github.com/ggerganov/llama.cpp
#
# How to run:
#   python convert_to_gguf.py
# ============================================================


import os           # For file/folder operations
import sys          # For running shell commands and exiting on errors
import subprocess   # For calling the llama.cpp conversion script
import torch        # PyTorch — needed to load the model

from transformers import AutoTokenizer, AutoModelForCausalLM  # Load model/tokenizer
from peft import PeftModel                                     # Load LoRA adapter


# ============================================================
# CONFIGURATION — Edit these paths to match your setup
# ============================================================

# The original base model used during training
BASE_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

# The folder where your LoRA adapter was saved by train_lora.py
LORA_MODEL_DIR = "./lora_output"

# Where to save the MERGED (combined base + LoRA) model before conversion
# This is a temporary folder — you can delete it after the GGUF is created
MERGED_MODEL_DIR = "./merged_model"

# ⚠️  IMPORTANT: Set this to the path where you cloned llama.cpp on your machine
# Example on Windows:  "C:/Users/YourName/llama.cpp"
# Example on Linux:    "/home/yourname/llama.cpp"
# Example on Mac:      "/Users/yourname/llama.cpp"
LLAMA_CPP_PATH = "/path/to/llama.cpp"  # <-- CHANGE THIS

# Output GGUF file name and location
# The final .gguf file will be saved here
OUTPUT_GGUF_FILE = "./smollm2_lora_finetuned.gguf"

# Quantization type — this controls the size vs quality tradeoff of the GGUF file.
# Common options:
#   "f16"   — Full 16-bit precision. Largest file, best quality. (~700MB for this model)
#   "q8_0"  — 8-bit quantization. Good quality, smaller file. (~380MB)
#   "q4_k_m" — 4-bit quantization. Smaller file, slight quality loss. (~200MB) ← recommended
#   "q4_0"  — 4-bit quantization (older method). Similar size to q4_k_m.
# For beginners, "q4_k_m" is a great balance of size and quality.
QUANTIZATION = "q4_k_m"


# ============================================================
# STEP 1: VALIDATE PATHS BEFORE DOING ANYTHING
# ============================================================

print("=" * 60)
print("GGUF Conversion Script for SmolLM2-360M + LoRA")
print("=" * 60)

# Check that the LoRA output folder exists
if not os.path.exists(LORA_MODEL_DIR):
    print(f"\n❌ ERROR: LoRA model folder '{LORA_MODEL_DIR}' not found.")
    print("   Please run train_lora.py first to generate your trained adapter.")
    sys.exit(1)

# Check that the adapter_config.json exists inside the folder
if not os.path.exists(os.path.join(LORA_MODEL_DIR, "adapter_config.json")):
    print(f"\n❌ ERROR: No adapter found inside '{LORA_MODEL_DIR}'.")
    print("   The folder exists but doesn't contain a valid LoRA adapter.")
    print("   Please re-run train_lora.py to generate the adapter.")
    sys.exit(1)

# Check that llama.cpp path is set and exists
if LLAMA_CPP_PATH == "/path/to/llama.cpp":
    print("\n❌ ERROR: You haven't set LLAMA_CPP_PATH yet!")
    print("   Open this script and change LLAMA_CPP_PATH to where you cloned llama.cpp.")
    print("   Example: LLAMA_CPP_PATH = '/home/yourname/llama.cpp'")
    sys.exit(1)

if not os.path.exists(LLAMA_CPP_PATH):
    print(f"\n❌ ERROR: llama.cpp folder not found at: '{LLAMA_CPP_PATH}'")
    print("   Clone it with: git clone https://github.com/ggerganov/llama.cpp")
    sys.exit(1)

# The conversion script inside llama.cpp we'll use
# llama.cpp provides a Python script to convert Hugging Face models to GGUF
convert_script = os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py")

if not os.path.exists(convert_script):
    print(f"\n❌ ERROR: Conversion script not found at: '{convert_script}'")
    print("   Make sure you have a recent version of llama.cpp.")
    print("   Try: git pull  inside your llama.cpp folder to update it.")
    sys.exit(1)

print("\n✅ All paths validated. Ready to proceed.\n")


# ============================================================
# STEP 2: LOAD BASE MODEL + LORA ADAPTER
# ============================================================

print("Loading base model and LoRA adapter...")
print(f"  Base model : {BASE_MODEL_NAME}")
print(f"  LoRA adapter: {LORA_MODEL_DIR}")

# Load the tokenizer from your LoRA output folder (it was saved there during training)
tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_DIR)

# Load the original base model weights
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map="cpu"            # Force everything onto CPU
)

# Load your trained LoRA adapter on top of the base model
# is_trainable=False because we're not training anymore — just merging
model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL_DIR,
    is_trainable=False
)

print("✅ Base model and LoRA adapter loaded.")


# ============================================================
# STEP 3: MERGE LORA ADAPTER INTO THE BASE MODEL
# ============================================================

print("\nMerging LoRA adapter into base model...")
print("(This bakes your training permanently into the model weights)")

# merge_and_unload() does two things:
#   1. MERGE: Mathematically adds the LoRA adapter weights into the base model weights
#   2. UNLOAD: Removes the LoRA wrapper so we get a plain, standard model back
# The result is a regular model that looks just like the original,
# but with your fine-tuned knowledge baked in.
merged_model = model.merge_and_unload()

print("✅ LoRA adapter successfully merged into base model.")


# ============================================================
# STEP 4: SAVE THE MERGED MODEL TO DISK
# ============================================================

print(f"\nSaving merged model to '{MERGED_MODEL_DIR}'...")
print("(This is needed before GGUF conversion — llama.cpp reads from a folder)")

# Create the output folder if it doesn't exist
os.makedirs(MERGED_MODEL_DIR, exist_ok=True)

# Save the merged model weights and config files
merged_model.save_pretrained(MERGED_MODEL_DIR)

# Save the tokenizer too — llama.cpp needs it for the GGUF file
tokenizer.save_pretrained(MERGED_MODEL_DIR)

print(f"✅ Merged model saved to '{MERGED_MODEL_DIR}'.")


# ============================================================
# STEP 5: CONVERT TO GGUF USING LLAMA.CPP
# ============================================================

print(f"\nConverting to GGUF format...")
print(f"  Quantization : {QUANTIZATION}")
print(f"  Output file  : {OUTPUT_GGUF_FILE}")
print("(This may take a few minutes...)\n")

# Build the command to run llama.cpp's Python conversion script.
# We call it using the same Python interpreter that's running this script (sys.executable),
# so it uses the same environment and installed packages.
convert_command = [
    sys.executable,         # e.g., "python" or "python3" — whatever is running this script
    convert_script,         # Path to llama.cpp's convert_hf_to_gguf.py
    MERGED_MODEL_DIR,       # Input: the folder with our merged model
    "--outtype", QUANTIZATION,  # Quantization format (e.g., "q4_k_m")
    "--outfile", OUTPUT_GGUF_FILE  # Output: where to save the .gguf file
]

print(f"Running command:\n  {' '.join(convert_command)}\n")

# subprocess.run() executes the command in the terminal.
# check=True means it will raise an error if the command fails.
try:
    result = subprocess.run(convert_command, check=True)
except subprocess.CalledProcessError as e:
    print(f"\n❌ ERROR: GGUF conversion failed!")
    print(f"   The conversion script returned an error.")
    print(f"   Common fixes:")
    print(f"   - Make sure llama.cpp is up to date (git pull)")
    print(f"   - Install llama.cpp Python deps: pip install -r {LLAMA_CPP_PATH}/requirements.txt")
    sys.exit(1)


# ============================================================
# STEP 6: DONE!
# ============================================================

print("\n" + "=" * 60)
print("✅ GGUF conversion complete!")
print(f"   Output file: {OUTPUT_GGUF_FILE}")

# Show file size so the user knows it worked
if os.path.exists(OUTPUT_GGUF_FILE):
    size_mb = os.path.getsize(OUTPUT_GGUF_FILE) / (1024 * 1024)
    print(f"   File size  : {size_mb:.1f} MB")

print("=" * 60)
print("\nYou can now use this GGUF file with:")
print("  • llama.cpp  → ./llama-cli -m smollm2_lora_finetuned.gguf -i")
print("  • Ollama     → ollama create mymodel -f Modelfile  (point Modelfile to the .gguf)")
print("  • LM Studio  → Load the .gguf file directly in the UI")
print()

# Optional cleanup reminder
print("💡 TIP: You can delete the './merged_model' folder to save disk space.")
print("        The .gguf file is all you need going forward.")
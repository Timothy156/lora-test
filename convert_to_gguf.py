# ============================================================
# CONVERT TRAINED LORA MODEL TO GGUF FORMAT
# ============================================================
# What is this file?
#   After training with train_lora.py, this script:
#     1. Detects whether you have a GPU or CPU
#     2. Loads your base model + LoRA adapter
#     3. MERGES them into a single standalone model
#     4. Converts that merged model into GGUF format
#
# Requirements:
#   pip install transformers peft torch
#   git clone https://github.com/ggerganov/llama.cpp
#
# How to run:
#   python convert_to_gguf.py
# ============================================================


import os           # File and folder operations
import sys          # For exiting on errors and reading the Python path
import subprocess   # For running the llama.cpp converter
import tempfile     # For creating the temporary wrapper script
import torch        # PyTorch — needed to load and merge the model

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# STEP 1: DETECT GPU OR CPU
# ============================================================

HAS_GPU = torch.cuda.is_available()

if HAS_GPU:
    GPU_NAME    = torch.cuda.get_device_name(0)
    GPU_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print("=" * 60)
    print(f"GPU detected: {GPU_NAME} ({GPU_VRAM_GB:.1f} GB VRAM)")
    print("   Merge step will use GPU (faster).")
    print("=" * 60)
    DEVICE_MAP  = "auto"
    MODEL_DTYPE = torch.float16
else:
    GPU_NAME = None
    print("=" * 60)
    print("No GPU detected — using CPU for merge step.")
    print("=" * 60)
    DEVICE_MAP  = "cpu"
    MODEL_DTYPE = torch.float32


# ============================================================
# CONFIGURATION — Edit these to match your setup
# ============================================================

BASE_MODEL_NAME  = "HuggingFaceTB/SmolLM2-360M"
LORA_MODEL_DIR   = "./lora_output"
MERGED_MODEL_DIR = "./merged_model"

# Set this to wherever you cloned llama.cpp
# Kaggle example : "/kaggle/working/llama.cpp"
# Linux example  : "/home/yourname/llama.cpp"
LLAMA_CPP_PATH = "/path/to/llama.cpp"   # <-- CHANGE THIS

OUTPUT_GGUF_FILE = "./smollm2_lora_finetuned.gguf"

# Valid --outtype values for this version of llama.cpp:
#   "f32"   — 32-bit full precision  (~1.4 GB)
#   "f16"   — 16-bit half precision  (~700 MB)
#   "bf16"  — bfloat16 precision     (~700 MB)
#   "q8_0"  — 8-bit quantization     (~380 MB)  <- recommended
#   "auto"  — let llama.cpp decide automatically
# NOTE: q4_k_m is NOT valid here. See tip at the bottom for how to get it.
QUANTIZATION = "f16"


# ============================================================
# STEP 2: VALIDATE ALL PATHS
# ============================================================

print("\n" + "=" * 60)
print("GGUF Conversion Script for SmolLM2-360M + LoRA")
print("=" * 60)

if not os.path.exists(LORA_MODEL_DIR):
    print(f"\nERROR: LoRA folder '{LORA_MODEL_DIR}' not found.")
    print("   Run train_lora.py first.")
    sys.exit(1)

if not os.path.exists(os.path.join(LORA_MODEL_DIR, "adapter_config.json")):
    print(f"\nERROR: No adapter found inside '{LORA_MODEL_DIR}'.")
    print("   Re-run train_lora.py to generate the adapter.")
    sys.exit(1)

if LLAMA_CPP_PATH == "/path/to/llama.cpp":
    print("\nERROR: You haven't set LLAMA_CPP_PATH!")
    print("   Open this script and change LLAMA_CPP_PATH to your llama.cpp folder.")
    sys.exit(1)

if not os.path.exists(LLAMA_CPP_PATH):
    print(f"\nERROR: llama.cpp not found at: '{LLAMA_CPP_PATH}'")
    print("   Clone it: git clone https://github.com/ggerganov/llama.cpp")
    sys.exit(1)

convert_script = os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py")
if not os.path.exists(convert_script):
    print(f"\nERROR: Conversion script not found at: '{convert_script}'")
    print("   Update llama.cpp: cd llama.cpp && git pull")
    sys.exit(1)

print("\nAll paths validated.\n")


# ============================================================
# STEP 3: LOAD BASE MODEL + LORA ADAPTER
# ============================================================

print("Loading base model and LoRA adapter...")
print(f"  Base model  : {BASE_MODEL_NAME}")
print(f"  LoRA adapter: {LORA_MODEL_DIR}")
print(f"  Device      : {'GPU (' + GPU_NAME + ')' if HAS_GPU else 'CPU'}")

tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_DIR)

# Load base model with float16 fallback for older GPUs
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=MODEL_DTYPE,
        device_map=DEVICE_MAP
    )
except Exception as e:
    if HAS_GPU:
        print(f"\n   Warning: float16 failed ({e}). Falling back to float32...")
        MODEL_DTYPE = torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map=DEVICE_MAP
        )
    else:
        raise

# Load LoRA adapter (is_trainable=False — we are just merging, not training)
model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR, is_trainable=False)
print("Base model and LoRA adapter loaded.")


# ============================================================
# STEP 4: MERGE LORA INTO BASE MODEL
# ============================================================

print("\nMerging LoRA adapter into base model...")

# merge_and_unload() permanently bakes the LoRA weights into the base model
# and removes the LoRA wrapper, leaving a single plain model.
merged_model = model.merge_and_unload()

# Always move to CPU + float32 before saving.
# llama.cpp's converter needs a float32 CPU model as input — it handles
# quantization itself. GPU/float16 was only used above for merge speed.
print("   Converting merged model to float32 on CPU for saving...")
merged_model = merged_model.to("cpu").to(torch.float32)

print("Merge complete.")


# ============================================================
# STEP 5: SAVE THE MERGED MODEL
# ============================================================

print(f"\nSaving merged model to '{MERGED_MODEL_DIR}'...")
os.makedirs(MERGED_MODEL_DIR, exist_ok=True)
merged_model.save_pretrained(MERGED_MODEL_DIR)
tokenizer.save_pretrained(MERGED_MODEL_DIR)
print("Merged model saved.")


# ============================================================
# STEP 6: CONVERT TO GGUF
# ============================================================

print(f"\nConverting to GGUF format...")
print(f"  Quantization : {QUANTIZATION}")
print(f"  Output file  : {OUTPUT_GGUF_FILE}")
print("  (This may take a few minutes...)\n")

# The arguments we want to pass to convert_hf_to_gguf.py.
# NO --vocab-type flag — it doesn't exist in this version (FIX 1).
# The converter auto-detects the tokenizer from tokenizer.json.
convert_args = [
    convert_script,
    MERGED_MODEL_DIR,
    "--outtype", QUANTIZATION,
    "--outfile", OUTPUT_GGUF_FILE,
]

# -------------------------------------------------------
# FIX 2: Torchvision circular import — proper fix
# -------------------------------------------------------
# The crash chain is:
#   llama.cpp imports transformers
#   → transformers checks if torchvision is available
#   → importlib.util.find_spec('torchvision') reads torchvision.__spec__
#   → our old dummy module had __spec__ = None → ValueError crash
#
# WHY MagicMock() works:
#   MagicMock() is a special test object that returns a valid mock for
#   EVERY attribute access automatically — including __spec__, __version__,
#   __path__, and anything else Python or transformers might read.
#   So when transformers does: sys.modules['torchvision'].__spec__
#   it gets back another MagicMock() instead of crashing on None.
#
# HOW we apply it:
#   We write a tiny wrapper Python script to a temp file.
#   That wrapper installs the MagicMocks into sys.modules FIRST,
#   before any other import runs, then calls the real converter.
#   Because it runs as a separate subprocess, it has a clean sys.modules
#   with no torchvision loaded yet — so our mock slots in perfectly.
# -------------------------------------------------------

# Build the wrapper script as a plain string.
#
# WHY WE PATCH importlib.util.find_spec INSTEAD OF sys.modules:
#
# Previous attempts put MagicMock() or dummy modules into sys.modules['torchvision'].
# Both failed because importlib.util.find_spec (Python 3.12) does this internally:
#
#   if name in sys.modules:
#       spec = sys.modules[name].__spec__   # reads __spec__
#       if spec is None: raise ValueError   # crash if None
#       if not set: raise ValueError        # crash if unset/wrong type
#
# No matter what we put in sys.modules, find_spec validates __spec__ strictly.
#
# THE REAL FIX: patch find_spec itself BEFORE any import runs.
# When our patched version sees "torchvision", it immediately returns None.
# This tells transformers "package not found" — it sets is_torchvision_available=False
# and moves on without ever touching sys.modules or crashing.
#
# We also set sys.modules['torchvision'] = None as a backup, which causes a clean
# ImportError (not a crash) if anything tries to actually do "import torchvision".

wrapper_code = "\n".join([
    "import sys",
    "import importlib.util",
    "",
    "# --- DEFINITIVE torchvision block ---",
    "# Step 1: Patch find_spec to return None for torchvision.",
    "# This intercepts the is_torchvision_available() check in transformers",
    "# BEFORE it ever touches sys.modules, avoiding all __spec__ validation.",
    "_orig_find_spec = importlib.util.find_spec",
    "def _patched_find_spec(name, package=None):",
    "    if name and (name == 'torchvision' or name.startswith('torchvision.')):",
    "        return None  # pretend torchvision doesn't exist",
    "    return _orig_find_spec(name, package)",
    "importlib.util.find_spec = _patched_find_spec",
    "",
    "# Step 2: Also block sys.modules so any stray 'import torchvision'",
    "# gets a clean ImportError instead of the circular import crash.",
    "# Setting a key to None in sys.modules is the official Python way",
    "# to make a module 'not importable' without actually removing it.",
    "for _m in ['torchvision','torchvision.transforms','torchvision._meta_registrations',",
    "           'torchvision.extension','torchvision.datasets','torchvision.models',",
    "           'torchvision.ops','torchvision.utils','torchvision.io']:",
    "    sys.modules[_m] = None",
    "# --- end torchvision block ---",
    "",
    "# Set sys.argv so convert_hf_to_gguf.py sees the right arguments",
    f"sys.argv = {repr(convert_args)}",
    "",
    "# Run the actual converter script as if called directly from the command line",
    "import runpy",
    f"runpy.run_path({repr(convert_script)}, run_name='__main__')",
])

# Write the wrapper to a temp file on disk
tmp_wrapper = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
tmp_wrapper.write(wrapper_code)
tmp_wrapper.close()

print(f"   Wrapper script written to: {tmp_wrapper.name}")
print(f"   Running conversion...\n")

try:
    subprocess.run(
        [sys.executable, tmp_wrapper.name],
        check=True,
        env={
            **os.environ,
            "TOKENIZERS_PARALLELISM": "false",  # Prevent tokenizer deadlock in subprocess
        }
    )

except subprocess.CalledProcessError:
    print(f"\nERROR: GGUF conversion failed!")
    print(f"\n   Things to try:")
    print(f"   1. Update llama.cpp:  cd {LLAMA_CPP_PATH} && git pull")
    print(f"   2. Install its Python requirements:")
    print(f"      pip install -r {LLAMA_CPP_PATH}/requirements.txt")
    print(f"   3. Check '{MERGED_MODEL_DIR}' has: config.json, tokenizer.json, *.safetensors")
    sys.exit(1)

finally:
    # Always delete the temp wrapper file, even if conversion crashed
    if os.path.exists(tmp_wrapper.name):
        os.remove(tmp_wrapper.name)


# ============================================================
# STEP 7: DONE!
# ============================================================

print("\n" + "=" * 60)
print("GGUF conversion complete!")
print(f"   Output file : {OUTPUT_GGUF_FILE}")

if os.path.exists(OUTPUT_GGUF_FILE):
    size_mb = os.path.getsize(OUTPUT_GGUF_FILE) / (1024 * 1024)
    print(f"   File size   : {size_mb:.1f} MB")

print("=" * 60)
print("\nYou can now use this GGUF file with:")
print("  llama.cpp  ->  ./llama-cli -m smollm2_lora_finetuned.gguf -i")
print("  Ollama     ->  Create a Modelfile pointing to the .gguf, then:")
print("                 ollama create mymodel -f Modelfile")
print("  LM Studio  ->  Load the .gguf directly in the UI")
print()
print("TIP: Delete './merged_model' folder to save disk space after conversion.")
print()
print("HOW TO GET q4_k_m (smaller, faster file):")
print("  This script outputs an f16 GGUF (~700 MB) as the intermediate format.")
print("  To compress it further to q4_k_m (~200 MB), run llama-quantize:")
print(f"    {LLAMA_CPP_PATH}/llama-quantize {OUTPUT_GGUF_FILE} ./smollm2_q4km.gguf q4_k_m")
print()
print("WHY TWO STEPS?")
print("  llama-quantize can only read f16/f32 source files.")
print("  If you try to quantize an already-quantized file (e.g. q8_0),")
print("  you get: 'requantizing from type q8_0 is disabled'")
print("  So: convert to f16 first (this script), then quantize (llama-quantize).")

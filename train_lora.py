# ============================================================
# LORA FINE-TUNING SCRIPT FOR SmolLM2-360M (AUTO CPU/GPU)
# ============================================================
# What is this file?
#   This script trains a small AI language model (SmolLM2-360M)
#   using a technique called LoRA (Low-Rank Adaptation).
#   It automatically detects whether you have a GPU and adjusts
#   its settings to get the best performance on your hardware.
#
# What is LoRA?
#   Instead of retraining the ENTIRE model (which takes huge
#   resources), LoRA only trains a small set of extra "adapter"
#   weights. This is much faster and uses less memory.
#
# Requirements (install these first by running in your terminal):
#   pip install transformers peft datasets pandas torch accelerate
# ============================================================


import os           # For file and folder operations
import pandas as pd # For reading your CSV file
import torch        # PyTorch — the core deep learning library
from peft import PeftModel

# Hugging Face Transformers — loads pretrained AI models
from transformers import (
    AutoTokenizer,          # Converts text into numbers the model understands
    AutoModelForCausalLM,   # Loads the actual language model
    TrainingArguments,      # Settings/config for the training process
    Trainer,                # The object that handles the training loop
    DataCollatorForSeq2Seq  # Batches and pads sequences during training
)

# PEFT = Parameter-Efficient Fine-Tuning (the library that handles LoRA)
from peft import (
    get_peft_model,     # Wraps your model with LoRA adapters
    LoraConfig,         # The settings/config for LoRA
    TaskType            # Tells LoRA what kind of task we're doing
)

# Hugging Face Datasets — a library to work with data for AI training
from datasets import Dataset


# ============================================================
# STEP 1: DETECT GPU OR CPU
# ============================================================
# torch.cuda.is_available() checks if your machine has an NVIDIA GPU
# with the correct CUDA drivers installed.
# If it returns True  → we use the GPU (faster training)
# If it returns False → we fall back to CPU automatically

HAS_GPU = torch.cuda.is_available()

if HAS_GPU:
    # torch.cuda.get_device_name(0) returns the name of your first GPU
    # e.g. "NVIDIA GeForce RTX 3080"
    GPU_NAME = torch.cuda.get_device_name(0)

    # total_memory is in bytes, so we divide to convert to gigabytes (GB)
    GPU_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print("=" * 60)
    print("🎮 GPU DETECTED — Using GPU for training!")
    print(f"   GPU  : {GPU_NAME}")
    print(f"   VRAM : {GPU_VRAM_GB:.1f} GB")
    print("=" * 60)
else:
    GPU_NAME = None
    GPU_VRAM_GB = 0
    print("=" * 60)
    print("💻 No GPU detected — Using CPU for training.")
    print("   (Training will be slower, but everything will still work!)")
    print("=" * 60)


# ============================================================
# STEP 2: CONFIGURATION — Values auto-adjust based on hardware
# ============================================================

# --- Model Settings ---
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"  # The base model we start from
OUTPUT_DIR = "./lora_output"                 # Folder where trained model is saved

# --- Dataset Settings ---
CSV_FILE = "dataset.csv"                # Your CSV file name (same folder as script)
USER_COLUMN = "user inquery"            # Column name for user messages in your CSV
ASSISTANT_COLUMN = "assistant response" # Column name for assistant replies in your CSV

# --- LoRA Settings ---
# These stay the same whether you're on GPU or CPU.

# r = the "rank" of LoRA. Higher = more parameters trained = smarter but slower.
# 8 is a good beginner balance between speed and quality.
LORA_R = 8

# lora_alpha = a scaling factor. Usually set to 2x the rank (so 16 here).
LORA_ALPHA = 16

# lora_dropout = randomly disables some neurons during training.
# This prevents the model from just "memorizing" your data. 0.05 = 5%.
LORA_DROPOUT = 0.05

# --- Training Settings (AUTOMATICALLY different for GPU vs CPU) ---
if HAS_GPU:
    # -------------------------------------------------------
    # GPU SETTINGS — more aggressive since GPU is much faster
    # -------------------------------------------------------
    NUM_EPOCHS = 3          # How many full passes through your dataset
    BATCH_SIZE = 4          # GPU can process 4 examples at once (much faster)
    LEARNING_RATE = 2e-4    # Slightly lower LR works well with larger batches on GPU
    MAX_LENGTH = 512        # GPU has more memory, so we can handle longer text
    SAVE_STEPS = 50         # Save a checkpoint every 50 steps
    LOGGING_STEPS = 10      # Print progress every 10 steps
    GRAD_ACCUM_STEPS = 2    # Fewer accumulation steps needed (batch is already bigger)

    # torch.float16 = "half precision" — uses HALF the memory of float32.
    # GPUs have dedicated hardware for float16 math, making it much faster.
    MODEL_DTYPE = torch.float16
    USE_FP16 = True         # Enable 16-bit training (GPU only feature)

    # "auto" lets PyTorch automatically decide which device each layer goes on.
    # With a single GPU, everything goes on the GPU.
    DEVICE_MAP = "auto"

    # use_cpu=False tells the Trainer NOT to force CPU mode
    USE_CPU = False

else:
    # -------------------------------------------------------
    # CPU SETTINGS — conservative to avoid slow speeds / crashes
    # -------------------------------------------------------
    NUM_EPOCHS = 3          # Same number of passes
    BATCH_SIZE = 1          # CPU can only safely handle 1 example at a time
    LEARNING_RATE = 3e-4    # Slightly higher LR to compensate for smaller batch
    MAX_LENGTH = 256        # Shorter sequences = less RAM used
    SAVE_STEPS = 50         # Save a checkpoint every 50 steps
    LOGGING_STEPS = 1       # Log EVERY step — useful since CPU training is slow
    GRAD_ACCUM_STEPS = 4    # Accumulate 4 steps to simulate a batch size of 4

    # torch.float32 = "full precision" — the standard for CPU.
    # CPUs don't have special hardware for float16, so we must use float32.
    MODEL_DTYPE = torch.float32
    USE_FP16 = False        # DO NOT use 16-bit on CPU — it will cause errors

    # "cpu" forces the model to stay on the CPU entirely
    DEVICE_MAP = "cpu"

    # use_cpu=True tells the Trainer to stay on CPU mode
    USE_CPU = True


# Print a clear summary so you know exactly what settings are being used
print(f"\n📋 Training Configuration Summary:")
print(f"   Device        : {'GPU (' + GPU_NAME + ')' if HAS_GPU else 'CPU'}")
print(f"   Epochs        : {NUM_EPOCHS}")
print(f"   Batch size    : {BATCH_SIZE}")
print(f"   Learning rate : {LEARNING_RATE}")
print(f"   Max length    : {MAX_LENGTH} tokens")
print(f"   Precision     : {'float16 — faster, less memory (GPU)' if USE_FP16 else 'float32 — standard (CPU)'}")
print(f"   Grad accum    : {GRAD_ACCUM_STEPS} steps (effective batch = {BATCH_SIZE * GRAD_ACCUM_STEPS})")
print()


# ============================================================
# STEP 3: LOAD THE TOKENIZER AND MODEL
# ============================================================

print("=" * 60)
print("Loading tokenizer and model...")
print("(This may take a few minutes on first run — it downloads the model)")
print("=" * 60)

# The tokenizer converts raw text into numbers the model understands,
# and converts numbers back to text for the output.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set the padding token to the end-of-sequence token if one isn't defined.
# This is required so we can batch multiple examples together.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Check if we have saved adapter weights from a previous training run.
# If so, load those instead of starting from scratch.
if os.path.exists(OUTPUT_DIR) and os.path.exists(os.path.join(OUTPUT_DIR, "adapter_config.json")):
    print("Found previous training! Resuming from saved adapter...")

    # -------------------------------------------------------
    # CROSS-DEVICE SAFE LOADING
    # -------------------------------------------------------
    # When resuming, we ALWAYS load the base model in float32 first,
    # regardless of whether we're on CPU or GPU.
    #
    # Why? Because the saved adapter from a previous run might have been
    # trained on a different device:
    #   - CPU training saves weights as float32
    #   - GPU training saves weights as float16
    # If we loaded the base model as float16 but the adapter is float32
    # (or vice versa), PyTorch would throw a dtype mismatch error.
    #
    # The safe solution: load EVERYTHING as float32 first,
    # then convert the whole combined model to the right dtype for THIS device.
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,    # Always float32 when resuming — safe for any device
        device_map=DEVICE_MAP   # "auto" for GPU, "cpu" for CPU
    )

    # Load your saved LoRA adapter on top of the base model.
    # is_trainable=True means we want to KEEP TRAINING (not just run inference).
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR, is_trainable=True)

    # NOW convert everything to the correct dtype for this device.
    # .to(MODEL_DTYPE) shifts all weights to float16 (GPU) or keeps float32 (CPU).
    # We do this AFTER loading so the conversion is applied to both
    # the base model AND the adapter weights together, at the same time.
    model = model.to(MODEL_DTYPE)

    print(f"   Adapter loaded and converted to {'float16 (GPU)' if HAS_GPU else 'float32 (CPU)'}.")
    print(f"   Safe to resume even if previous training was on a different device!")

else:
    print("No previous training found. Starting fresh...")

    # Fresh start — load the base model directly in the correct dtype for this device
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=MODEL_DTYPE,      # float16 for GPU, float32 for CPU
        device_map=DEVICE_MAP   # "auto" for GPU, "cpu" for CPU
    )

print(f"\n✅ Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# STEP 4: APPLY LORA TO THE MODEL
# ============================================================

print("\nApplying LoRA adapters to the model...")

# LoraConfig defines the structure of the LoRA adapters
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # We're doing text generation (causal LM)
    r=LORA_R,                        # Rank — how many extra parameters to train
    lora_alpha=LORA_ALPHA,           # Scaling factor for the adapter output
    lora_dropout=LORA_DROPOUT,       # Dropout to prevent overfitting

    # target_modules: The attention layers inside the model that get LoRA adapters.
    # "q_proj" = query projection, "v_proj" = value projection.
    # These are the most impactful layers to fine-tune in transformer models.
    target_modules=["q_proj", "v_proj"],

    bias="none",  # Don't train bias terms — keeps the adapter small and fast
)

# Wrap the base model with LoRA adapters.
# After this line, ONLY the tiny LoRA layers will be updated during training.
model = get_peft_model(model, lora_config)

# This prints exactly how many parameters are being trained vs. the total.
# You'll see it's a very small fraction — that's the whole point of LoRA!
model.print_trainable_parameters()


# ============================================================
# STEP 5: LOAD AND PREPARE YOUR DATASET
# ============================================================

print(f"\nLoading dataset from '{CSV_FILE}'...")

try:
    # pd.read_csv() reads your spreadsheet into a Python table (DataFrame)
    df = pd.read_csv(CSV_FILE)
    print(f"✅ Loaded {len(df)} rows from CSV.")
    print(f"   Columns found: {list(df.columns)}")
except FileNotFoundError:
    print(f"\n❌ ERROR: Could not find '{CSV_FILE}'.")
    print("   Please make sure your CSV file is in the same folder as this script.")
    print(f"   The CSV must have columns: '{USER_COLUMN}' and '{ASSISTANT_COLUMN}'")
    exit(1)

# Make sure both required columns exist in the file
if USER_COLUMN not in df.columns or ASSISTANT_COLUMN not in df.columns:
    print(f"\n❌ ERROR: CSV must have columns named '{USER_COLUMN}' and '{ASSISTANT_COLUMN}'.")
    print(f"   Found columns: {list(df.columns)}")
    exit(1)

# Remove rows where either the question or answer is blank/missing
df = df.dropna(subset=[USER_COLUMN, ASSISTANT_COLUMN])
print(f"   Rows after removing empty entries: {len(df)}")


def format_prompt(row):
    """
    Formats one CSV row into a conversation string.
    The model learns to generate the 'Assistant:' part when given the 'User:' part.
    The eos_token (<|endoftext|>) signals the end of the conversation.
    """
    return f"User: {row[USER_COLUMN]}\nAssistant: {row[ASSISTANT_COLUMN]}{tokenizer.eos_token}"


# Apply the formatting to every row in the dataset
df["text"] = df.apply(format_prompt, axis=1)

# Show one example so you can confirm the format looks correct
print("\nSample formatted prompt:")
print("-" * 40)
print(df["text"].iloc[0])
print("-" * 40)


def tokenize_function(examples):
    """
    Converts text strings into lists of token IDs (numbers) for the model.
    - truncation=True  : cuts off text that's longer than MAX_LENGTH
    - padding="max_length" : pads shorter text so all examples are equal length
    - labels = input_ids : the model learns by predicting each next token
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors=None     # Return plain Python lists (not tensors yet)
    )

    # Labels are a copy of input_ids.
    # During training, the model tries to predict each token given all previous tokens.
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


# Convert the Pandas DataFrame into a Hugging Face Dataset object
hf_dataset = Dataset.from_pandas(df[["text"]])

# Tokenize the entire dataset (batched=True is faster than processing one-by-one)
tokenized_dataset = hf_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]     # Remove the raw text — we only need token IDs now
)

print(f"\n✅ Dataset tokenized! {len(tokenized_dataset)} training examples ready.")


# ============================================================
# STEP 6: CONFIGURE TRAINING SETTINGS
# ============================================================

print("\nSetting up training configuration...")

# All settings here use the GPU/CPU variables we defined in Step 2
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,                        # Where to save checkpoints and the final model
    num_train_epochs=NUM_EPOCHS,                  # Total passes through the data
    per_device_train_batch_size=BATCH_SIZE,       # Examples per batch (4 for GPU, 1 for CPU)
    gradient_accumulation_steps=GRAD_ACCUM_STEPS, # Steps to accumulate before updating weights

    learning_rate=LEARNING_RATE,                  # How large each weight update is
    save_steps=SAVE_STEPS,                        # Save a checkpoint every N steps
    logging_steps=LOGGING_STEPS,                  # Print a log every N steps
    save_total_limit=2,                           # Only keep the 2 most recent checkpoints

    use_cpu=USE_CPU,            # True = stay on CPU | False = use GPU
    fp16=USE_FP16,              # True = faster 16-bit math (GPU only) | False = standard 32-bit

    remove_unused_columns=False, # Keep all dataset columns (some used internally)
    report_to="none",            # Don't send data to Weights & Biases or similar tools
    push_to_hub=False,           # Don't upload to Hugging Face Hub
)


# ============================================================
# STEP 7: CREATE THE TRAINER AND START TRAINING
# ============================================================

# The Trainer automates the entire training loop:
# → feed data → compute loss → backpropagate → update weights → repeat
trainer = Trainer(
    model=model,                      # The LoRA-wrapped model to train
    args=training_args,               # All settings from above
    train_dataset=tokenized_dataset,  # Your prepared dataset
    processing_class=tokenizer,       # Handles padding/batching (renamed in newer transformers)
)

print("\n" + "=" * 60)
print("🚀 Starting training...")
print(f"   Device            : {'GPU (' + GPU_NAME + ')' if HAS_GPU else 'CPU'}")
print(f"   Epochs            : {NUM_EPOCHS}")
print(f"   Training examples : {len(tokenized_dataset)}")
print(f"   Max token length  : {MAX_LENGTH}")
print("=" * 60)

if HAS_GPU:
    print("(Training on GPU — this should be significantly faster! 🚀)")
else:
    print("(Training on CPU — this will be slow. Be patient! ☕)")
print()

# Run the training loop!
trainer.train()

print("\n✅ Training complete!")


# ============================================================
# STEP 8: SAVE THE TRAINED MODEL
# ============================================================

print(f"\nSaving trained model to '{OUTPUT_DIR}'...")

# Save the LoRA adapter weights (small file — just the trained adapter, not the full model)
model.save_pretrained(OUTPUT_DIR)

# Save the tokenizer alongside the model so everything stays together
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Model saved to '{OUTPUT_DIR}' folder.")
print("\nYou can now run 'test_lora.py' to chat with your trained model!")
print("=" * 60)

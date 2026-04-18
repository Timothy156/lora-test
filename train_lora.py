# ============================================================
# LORA FINE-TUNING SCRIPT FOR SmolLM2-360M (CPU ONLY)
# ============================================================
# What is this file?
#   This script trains a small AI language model (SmolLM2-360M)
#   using a technique called LoRA (Low-Rank Adaptation).
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
# STEP 1: CONFIGURATION — Set your settings here
# ============================================================

# --- Model Settings ---
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"  # The base model we start from
OUTPUT_DIR = "./lora_output"                 # Folder where trained model is saved

# --- Dataset Settings ---
CSV_FILE = "dataset.csv"                     # Your CSV file name (put it in the same folder)
USER_COLUMN = "user inquery"                 # The column name for user messages in your CSV
ASSISTANT_COLUMN = "assistant response"      # The column name for assistant replies in your CSV

# --- LoRA Settings ---
# r = the "rank" of LoRA. Higher = more parameters trained = smarter but slower.
# 8 is a good beginner balance between speed and quality.
LORA_R = 8

# lora_alpha = a scaling factor. Usually set to 2x the rank (so 16 here).
LORA_ALPHA = 16

# lora_dropout = randomly turns off some neurons during training to prevent
# the model from "memorizing" instead of "learning". 0.05 = 5% dropout.
LORA_DROPOUT = 0.05

# --- Training Settings ---
NUM_EPOCHS = 3          # How many times the model sees your full dataset
BATCH_SIZE = 2          # How many examples are processed at once (keep at 1 for CPU)
LEARNING_RATE = 3e-4    # How big each learning step is (3e-4 = 0.0003)
MAX_LENGTH = 256        # Maximum number of tokens (word pieces) per example
SAVE_STEPS = 50         # Save a checkpoint every N training steps
LOGGING_STEPS = 1      # Print progress every N training steps


# ============================================================
# STEP 2: LOAD THE TOKENIZER AND MODEL
# ============================================================

print("=" * 60)
print("Loading tokenizer and model...")
print("(This may take a few minutes on first run — it downloads the model)")
print("=" * 60)

# The tokenizer converts raw text like "Hello world" into a list of numbers
# that the model can process. It also does the reverse (numbers → text).
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Many tokenizers don't have a "padding token" by default.
# We set it to the end-of-sequence token so we can batch examples together.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the actual language model weights from Hugging Face.
# dtype=torch.float32 means we use 32-bit numbers (standard for CPU).
# Check if a previous trained adapter exists
if os.path.exists(OUTPUT_DIR) and os.path.exists(os.path.join(OUTPUT_DIR, "adapter_config.json")):
    print("Found previous training! Resuming from saved adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR, is_trainable=True)
else:
    print("No previous training found. Starting fresh...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

print(f"\n✅ Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# STEP 3: APPLY LORA TO THE MODEL
# ============================================================

print("\nApplying LoRA adapters to the model...")

# LoraConfig defines HOW we apply LoRA to the model.
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # We're doing causal language modeling (text generation)
    r=LORA_R,                       # Rank (see explanation above)
    lora_alpha=LORA_ALPHA,          # Scaling factor (see explanation above)
    lora_dropout=LORA_DROPOUT,      # Dropout rate (see explanation above)

    # target_modules: Which layers inside the model get LoRA adapters.
    # "q_proj" and "v_proj" are attention layers — the most important ones to adapt.
    target_modules=["q_proj", "v_proj"],

    bias="none",  # We don't train bias terms — keeps things simple
)

# Wrap the base model with LoRA. After this, only the LoRA layers will be trained.
model = get_peft_model(model, lora_config)

# Print how many parameters are being trained vs. total.
# You'll see it's a tiny fraction — that's the power of LoRA!
model.print_trainable_parameters()


# ============================================================
# STEP 4: LOAD AND PREPARE YOUR DATASET
# ============================================================

print(f"\nLoading dataset from '{CSV_FILE}'...")

# --- Read the CSV file ---
# Make sure your CSV file is in the same folder as this script.
# It should have two columns: one for user messages, one for assistant replies.
try:
    df = pd.read_csv(CSV_FILE)
    print(f"✅ Loaded {len(df)} rows from CSV.")
    print(f"   Columns found: {list(df.columns)}")
except FileNotFoundError:
    print(f"\n❌ ERROR: Could not find '{CSV_FILE}'.")
    print("   Please make sure your CSV file is in the same folder as this script.")
    print("   The CSV must have these columns:")
    print(f"   - '{USER_COLUMN}'")
    print(f"   - '{ASSISTANT_COLUMN}'")
    exit(1)

# Check that the expected columns exist
if USER_COLUMN not in df.columns or ASSISTANT_COLUMN not in df.columns:
    print(f"\n❌ ERROR: CSV must have columns named '{USER_COLUMN}' and '{ASSISTANT_COLUMN}'.")
    print(f"   Found columns: {list(df.columns)}")
    exit(1)

# Drop any rows that have empty/missing values
df = df.dropna(subset=[USER_COLUMN, ASSISTANT_COLUMN])
print(f"   Rows after removing empty entries: {len(df)}")


# --- Format each row into a prompt ---
# We combine the user question and assistant answer into one text string.
# The model learns to generate the assistant's response given the user's input.
def format_prompt(row):
    """
    Takes one row from the CSV and formats it as a conversation.
    The model will learn to produce the 'Response' part given the 'User:' part.
    """
    return f"User: {row[USER_COLUMN]}\nAssistant: {row[ASSISTANT_COLUMN]}{tokenizer.eos_token}"


# Apply formatting to every row in the dataset
df["text"] = df.apply(format_prompt, axis=1)

# Show a sample so you can verify it looks right
print("\nSample formatted prompt:")
print("-" * 40)
print(df["text"].iloc[0])
print("-" * 40)


# --- Tokenize the dataset ---
# Convert each formatted text string into a list of token IDs (numbers).
def tokenize_function(examples):
    """
    Converts raw text into token IDs.
    truncation=True cuts off text that's too long.
    padding="max_length" pads short texts so all examples are the same length.
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors=None  # Return plain Python lists, not tensors yet
    )

    # For causal language modeling, the "labels" are the same as the input IDs.
    # The model learns to predict the next token in the sequence.
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


# Convert our Pandas DataFrame into a Hugging Face Dataset object
hf_dataset = Dataset.from_pandas(df[["text"]])

# Apply tokenization to the entire dataset
# batched=True processes multiple examples at once for speed
tokenized_dataset = hf_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # Remove the raw text column — we only need token IDs
)

print(f"\n✅ Dataset tokenized! {len(tokenized_dataset)} training examples ready.")


# ============================================================
# STEP 5: CONFIGURE TRAINING SETTINGS
# ============================================================

print("\nSetting up training configuration...")

# TrainingArguments holds all the hyperparameters for training.
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,              # Where to save checkpoints and the final model
    num_train_epochs=NUM_EPOCHS,        # Number of full passes through your dataset
    per_device_train_batch_size=BATCH_SIZE,  # Examples per batch (keep at 1 for CPU)

    # Gradient accumulation: since our batch size is 1, we accumulate gradients
    # over 4 steps before updating weights. Effectively simulates a batch size of 4.
    gradient_accumulation_steps=4,

    learning_rate=LEARNING_RATE,        # Step size for weight updates
    save_steps=SAVE_STEPS,              # Save a checkpoint every N steps
    logging_steps=LOGGING_STEPS,        # Log loss/progress every N steps
    save_total_limit=2,                 # Keep only the 2 most recent checkpoints (saves disk space)

    # use_cpu=True forces training on CPU even if a GPU is available
    use_cpu=True,

    # fp16=False: Don't use 16-bit precision (CPU doesn't support it well)
    fp16=False,

    # remove_unused_columns=False: Keep all dataset columns (some may be needed internally)
    remove_unused_columns=False,

    # report_to="none": Don't send logs to Weights & Biases or other trackers
    report_to="none",

    # Disable push to Hugging Face Hub
    push_to_hub=False,
)


# ============================================================
# STEP 6: CREATE THE TRAINER AND START TRAINING
# ============================================================

# The Trainer handles the entire training loop:
# forward pass → calculate loss → backpropagation → update weights → repeat
trainer = Trainer(
    model=model,                     # The LoRA-wrapped model
    args=training_args,              # Training settings from above
    train_dataset=tokenized_dataset, # Your tokenized data
    processing_class=tokenizer,      # Replaces the old 'tokenizer=' argument in newer versions of transformers
)

print("\n" + "=" * 60)
print("🚀 Starting training...")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Training examples: {len(tokenized_dataset)}")
print(f"   Max token length: {MAX_LENGTH}")
print("=" * 60)
print("(Training on CPU will be slow. Be patient! ☕)")
print()

# This line actually runs the training!
trainer.train()

print("\n✅ Training complete!")


# ============================================================
# STEP 7: SAVE THE TRAINED MODEL
# ============================================================

print(f"\nSaving trained model to '{OUTPUT_DIR}'...")

# Save the LoRA adapter weights (NOT the full model — just the small adapter)
model.save_pretrained(OUTPUT_DIR)

# Save the tokenizer alongside the model so we can load it later
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Model saved to '{OUTPUT_DIR}' folder.")
print("\nYou can now run 'test_lora.py' to chat with your trained model!")
print("=" * 60)
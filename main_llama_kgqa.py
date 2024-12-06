from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, json, wandb
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
from SPARQLWrapper import SPARQLWrapper, JSON, N3
from huggingface_hub import login
from dotenv import load_dotenv


# QALD-9-plus files
train_data_file = "QALD_9_plus/data/qald_9_plus_train_dbpedia.json"
test_data_file = "QALD_9_plus/data/qald_9_plus_test_dbpedia.json"
filtred_train_data_file = "filtred_QALD_9plus/filtred_qald_9_plus_train_dbpedia.json"
filtred_test_data_file = "filtred_QALD_9plus/filtred_qald_9_plus_test_dbpedia.json"

# QALD-9-plus files
#train_data_file = "QALD_9_plus/data/qald_9_plus_train_wikidata.json"
#test_data_file = "QALD_9_plus/data/qald_9_plus_test_wikidata.json"
#filtred_train_data_file = "filtred_QALD_9plus/filtred_qald_9_plus_train_wikidata.json"
#filtred_test_data_file = "filtred_QALD_9plus/filtred_qald_9_plus_test_wikidata.json"

# QALD-10 files
#train_data_file = "QALD-10/data/qald_9_plus/qald_9_plus_train_wikidata.json"
#test_data_file = "QALD-10/data/qald_10/qald_10.json"
#filtred_train_data_file = "filtred_QALD_10/filtred_qald_10_train_dbpedia.json"
#filtred_test_data_file = "filtred_QALD_10/filtred_qald_10_test_dbpedia.json"


base_model = "../../.cacheThamer/Meta-Llama-3-8B-Instruct"
dataset_name = {"train": filtred_train_data_file, "test": filtred_test_data_file}

epochs_val = 6

MODEL_NAME = "Llama-KGQA"
RESULTS_DIR = MODEL_NAME + "/"
output_model_name = RESULTS_DIR + "llama_kgqa.llama.model"
fine_tuned_model_path = RESULTS_DIR + "fine_tuned.llama.model"


# Load environment variables from the .env file
load_dotenv()

# Access the token
hf_token = os.getenv("HUGGINGFACE_TOKEN")
wb_token = os.getenv("WANDB_TOKEN")

# Login into HF and WandB
login(token = hf_token)
wandb.login(key=wb_token)

run = wandb.init(
    project=MODEL_NAME,
    job_type="training",
    anonymous="allow"
)

torch_dtype = torch.float16
attn_implementation = "eager"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

model, tokenizer = setup_chat_format(model, tokenizer)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)

#Importing the dataset
def read_json(filename):
    with open(filename, 'r', encoding="utf8") as f:
        return json.load(f)


def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def filter_data(data):
    filtered_questions = []
    language_list = ['en']
    for q in data['questions']:
        question = {"id": q["id"],
                    "question": list(filter(lambda d: d['language'] in language_list, q['question']))[0]["string"],
                    "sparql": q["query"]["sparql"]}
        filtered_questions.append(question)
    json_dict = {"questions": filtered_questions}
    return json_dict


train = read_json(train_data_file)
test = read_json(test_data_file)

filtred_train_data = filter_data(train)
filtred_test_data = filter_data(test)

save_json(filtred_train_data_file, filtred_train_data)
save_json(filtred_test_data_file, filtred_test_data)

data_files = {"train": filtred_train_data_file, "test": filtred_test_data_file}
dataset = load_dataset("json", data_files=data_files, field="questions")


def format_chat_template(row):
    row_json = [{"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["sparql"] + "<|eot_id|>"}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row


dataset = dataset.map(
    format_chat_template,
    num_proc=4,
)

# Complaining and training the model
training_arguments = TrainingArguments(
    output_dir=output_model_name,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=epochs_val,
    evaluation_strategy="steps",#
    eval_steps=0.2,#
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length=512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False
)

trainer.train()

# Model evaluation
wandb.finish()
model.config.use_cache = True

# Save trained model
trainer.model.save_pretrained(fine_tuned_model_path)


# Merging the base model with the adapter
# Reload tokenizer and model
tokenizer_relod = AutoTokenizer.from_pretrained(base_model)

base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

base_model_reload, tokenizer_relod = setup_chat_format(base_model_reload, tokenizer_relod)

# Merge adapter with base model
final_model = PeftModel.from_pretrained(base_model_reload, fine_tuned_model_path)

final_model = final_model.merge_and_unload()

# Save and push the final model
final_model.save_pretrained(output_model_name)
tokenizer_relod.save_pretrained(output_model_name)

final_model.push_to_hub(MODEL_NAME, use_temp_dir=False)
tokenizer_relod.push_to_hub(MODEL_NAME, use_temp_dir=False)

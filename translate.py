from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import torch, sys
from trl import setup_chat_format


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Provide a question")
        sys.exit(-1)

    # Path to the fine-tuned model
    model_id = "./Llama-KGQA/llama_kgqa.model"

    # Quantization config
    torch_dtype = torch.float16
    attn_implementation = "eager"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load the fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Define chat format
    model, tokenizer = setup_chat_format(model, tokenizer)

    messages = [
        {
            "role": "user",
            "content": sys.argv[1]
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(**inputs, num_return_sequences=1, eos_token_id=terminators, max_new_tokens=256,
                             do_sample=True, temperature=0.6, top_p=0.9)  # , max_length=150

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(text.split("assistant")[1])

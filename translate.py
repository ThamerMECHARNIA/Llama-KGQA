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
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

ENDPOINT = 'https://dbpedia.org/sparql'
WRONG_QUERY_MAX = 10


def querying(query):
    sparql = SPARQLWrapper(ENDPOINT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        qres = sparql.query().convert()
        if not qres.get("results", {}).get("bindings"):  # Check if results are empty
            return ("no_results", None)
        return ("success", qres)
    except SPARQLExceptions.QueryBadFormed as e:
        return ("syntax_error", str(e))
    except Exception as e:
        return ("error", str(e))


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

    wrong_queries_count = 0
    translation_ended = False
    while not translation_ended and wrong_queries_count < WRONG_QUERY_MAX:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(**inputs, num_return_sequences=1, eos_token_id=terminators, max_new_tokens=256,
                                 do_sample=True, temperature=0.6, top_p=0.9)  # , max_length=150

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        generated_query = text.split("assistant")[1]

        print(f"Generated Query {wrong_queries_count}:", generated_query)

        status, result = querying(generated_query)

        if status not in {"syntax_error", "error"}:
            translation_ended = True
        else:
            wrong_queries_count += 1

        if status == "success":
            print("Query succeeded with results:", result)
        elif status == "no_results":
            print("Query succeeded but returned no results.")
        elif status == "syntax_error":
            print("Query failed due to a syntax error:", result)
            if wrong_queries_count < WRONG_QUERY_MAX:
                print("Trying again ...")
        elif status == "error":
            print("Query failed due to an error:", result)
            if wrong_queries_count < WRONG_QUERY_MAX:
                print("Trying again ...")

# Llama-KGQA

**Llama-KGQA** is a fine-tuned model for question answering (QA) on knowledge graphs (KGs). The model was fine-tuned using the **QALD (Question Answering over Linked Data)** benchmark datasets ([QALD-9-plus](https://github.com/KGQA/QALD_9_plus) for Dbpedia and [QALD-10](https://github.com/KGQA/QALD-10) for Wikidata). This repository contains the results of testing different models, as well as scripts for fine-tuning and using the model to generate SPARQL queries from natural language (NL) questions.

## Repository Contents

### Results
The repository includes results for various models tested during fine-tuning and evaluation. The results pertain to the **QALD-9-plus-DBpedia** and **QALD-10-Wikidata** datasets:

- `Llama-2-7b_results.zip`: Results for the **Llama-2-7b** model.
- `Llama-3-8b_results.zip`: Results for the **Llama-3-8b** model.
- `Llama-3-70b_results.zip`: Results for the **Llama-3-70b** model.
- `Mixtral-7b_results.zip`: Results for the **Mixtral-7b** model.

Each directory contains detailed output.

### Scripts
This repository also includes the following Python scripts:

#### 1. **`main_llama_kgqa.py`**
This script is used to LoRA-fine-tune `Meta-Llama-3-8B-Instruct` model for SPARQL and créates `Llama-KGQA` model.

#### 2. **`translate.py`**
This script takes a natural language (NL) question as input and generates a SPARQL query using the fine-tuned model.

**Usage**:
```bash
python3 translate.py "[NATURAL_LANGUAGE_QUESTION]"
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


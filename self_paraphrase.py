import json
import os
import re
from collections import Counter
from typing import Dict, Optional, Tuple

import torch
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from tqdm import tqdm

from utils import dump_jsonl, load_jsonl


def extract_json(text: str) -> Optional[Dict]:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def validate_paraphrase(
    src: str,
    obj: Dict,
) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "Output is not a JSON object."

    if "paraphrase" not in obj or not isinstance(obj["paraphrase"], str):
        return False, "Missing string field: paraphrase."

    para = obj["paraphrase"].strip()

    if len(para) < 3:
        return False, "Paraphrase is too short."

    # Simple constraints you can hard-enforce:
    # 1) must not be identical
    if para == src.strip():
        return False, "Paraphrase is identical to source."

    # 2) preserve numbers (hard rule)
    src_nums = re.findall(r"\d+(?:\.\d+)?", src)
    para_nums = re.findall(r"\d+(?:\.\d+)?", para)
    if src_nums != para_nums:
        return False, f"Numbers not preserved. src={src_nums}, para={para_nums}"

    # 3) Assure [MASK] in the generation
    if "[MASK]" in src and "[MASK]" not in para:
        return False, "[MASK] token missing in paraphrase."
    
    return True, "OK"


class ParaphraseExtract(BaseModel):
    paraphrase: str = Field(
        description="A Boolean value indicating whether the sentence contains generalizable triple-like factual knowledge."
    )

def extract_first_json_block(text: str) -> str:
    """
    Extract the first valid JSON object from a string with nested braces.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No opening brace found in text")

    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return text[start:i+1]
    raise ValueError("Braces do not match, incomplete JSON block")

def robust_parse(output_str):
    """
    Preprocess LLM output string and try to parse it robustly using a LangChain parser.
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", output_str.strip(), flags=re.MULTILINE).strip()

    # Normalize casing for JSON literals
    cleaned = re.sub(r'\bNULL\b', 'null', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bTRUE\b', 'true', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bFALSE\b', 'false', cleaned, flags=re.IGNORECASE)

    try:
        json_block = extract_first_json_block(cleaned)
    except ValueError:
        return None
    
    try:
        return json_parser.parse(json_block).model_dump()
    except Exception:
        return None
    


def generate_text_batch(model, tokenizer, prompts: list[str], top_p, max_new_tokens, temperature) -> list[str]:
    """
    Generates text for a batch of prompts efficiently.
    """
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    input_len = inputs.input_ids.shape[1]
    generated_tokens = out[:, input_len:]
    decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return [text.strip() for text in decoded_texts]

SHARED_SYSTEM_RULES = (
    "You are a paraphrasing engine.\n"
    "You MUST follow the output contract exactly.\n"
    "Return ONLY valid JSON. No extra words.\n"
    "Strategies to apply:\n"
    "- Lexicon Replacement: Swap words for synonyms while maintaining nuance.\n"
    "- Syntax Reordering: Change grammatical structure (e.g., active/passive, clause order).\n"
    "- Semantic Restructuring: Express the same truth conditions using different wording logic (logical entailment).\n"
    "Constraints:\n"
    "- Maintain Strict Entailment: The paraphrase must be true if and only if the source is true.\n"
    "- Preserve ALL numbers/named entities exactly.\n"
    "- One sentence.\n"
    "- Do not add new facts.\n"
    "\nBelow are some examples: \n"
)

MYRIADLAMA_SYSTEM_RULES = (
    "For the given prompts: What do people use to absorb extra ink from a fountain pen?\n"
    "Lexicon Replacement paraphrase: In which continent can Queen Maud Land be found? [MASK]\n"
    "Syntax Reordering paraphrase: Queen Maud Land is a region located on [MASK] continent.\n"
    "Semantic Restructuring paraphrase: Identify the continental landmass that contains Queen Maud Land: [MASK]\n"
)

COMMONSENSEQA_SYSTEM_RULES = (
    "What do people use to absorb extra ink from a fountain pen?\n"
    "Lexicon Replacement paraphrase: What material do individuals utilize to soak up excess ink from a fountain pen?\n"
    "Syntax Reordering paraphrase: To absorb extra ink from a fountain pen, what is the item that is used?\n"
    "Semantic Restructuring paraphrase: Identify the object that serves the function of removing surplus ink from a fountain pen.\n"
)

MMLU_SYSTEM_RULES = (
    "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\n"
    "Lexicon Replacement paraphrase: Calculate the degree of the specific field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\n"
    "Syntax Reordering paraphrase: Over the field Q, what is the degree of the extension Q(sqrt(2), sqrt(3), sqrt(18))?\n"
    "Semantic Restructuring paraphrase: Determine the dimension of Q(sqrt(2), sqrt(3), sqrt(18)) when viewed as a vector space over Q.\n"
)

HOTPOTQA_SYSTEM_RULES = (
    "What American stage, film, and television actor who also appeared in a large number of musicals, played Samson in the 1949 film \"Samson and Delilah\".\n"
    "Lexicon Replacement paraphrase: Which US theater, movie, and TV performer, also featured in many musicals, portrayed the character Samson in the 1949 movie \"Samson and Delilah\"?\n"
    "Syntax Reordering paraphrase: In the 1949 film \"Samson and Delilah\", which American actor known for stage, screen, television, and numerous musicals was cast as Samson?\n"
    "Semantic Restructuring paraphrase: Identify the American entertainer with credits in stage, screen, television, and many musicals who starred as Samson in the 1949 motion picture \"Samson and Delilah\".\n"
)


def enforced_paraphrase(
    system_rules,
    prompt, 
    batch_size, 
    temperature, top_p,
    num_paraphrase, max_rounds, 
    verbose=False):
    
    contract = (
        'Output JSON schema:\n'
        '{\n'
        '  "paraphrase": string,\n'
        '}\n'
    )

    def build_prompt(user_msg: str) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_rules + "\n" + contract},
                {"role": "user", "content": user_msg},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return system_rules + "\n" + contract + "\nUSER:\n" + user_msg + "\nASSISTANT:\n"

    collected_paraphrases = []
    seen_texts = set()
    
    user_msg_base = f"Source: {prompt}\nTask: paraphrase the Source under the Constraints."
    query_text = build_prompt(user_msg_base)
    
    parse_error, valid_error, duplicated_error = 0, 0, 0
    valid_error_details = Counter()
    for round_idx in range(max_rounds):
        if len(collected_paraphrases) >= num_paraphrase:
            break
        
        current_batch_inputs = [query_text] * batch_size
        raw_outputs = generate_text_batch(
            model, 
            tokenizer, 
            current_batch_inputs, 
            top_p=top_p,
            max_new_tokens=256, 
            temperature=temperature
        )

        for raw in raw_outputs:
            if len(collected_paraphrases) >= num_paraphrase:
                break

            obj = robust_parse(raw)
            if not obj:
                parse_error += 1
                continue

            ok, reason = validate_paraphrase(prompt, obj)
            valid_error_details[reason] += 1
            if not ok:
                valid_error += 1
                continue
            candidate = obj["paraphrase"].strip()
            if candidate != prompt and candidate not in seen_texts:
                seen_texts.add(candidate)
                collected_paraphrases.append(candidate)
            else:
                duplicated_error += 1

    if verbose:
        print(f"Parsing errors: {parse_error}, Validation errors: {valid_error}, Duplicated paraphrases: {duplicated_error}")
        print("Validation error details:", dict(valid_error_details))
    return raw_outputs, collected_paraphrases[:num_paraphrase]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name to use for paraphrasing.")
    parser.add_argument("--model", type=str, required=True, help="Model name to use for paraphrasing.")
    parser.add_argument("--debug", action="store_true", help="Use debug mode with smaller dataset.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument("--num_paraphrase", type=int, default=2, help="Number of paraphrases to generate.")
    parser.add_argument("--max_rounds", type=int, default=2, help="Maximum rounds of paraphrasing.")
    parser.add_argument("--temperature", type=float, default=1.5, help="Temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling value for generation.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--rewrite", action="store_true", help="Rewrite existing paraphrase files.")
    args = parser.parse_args()    

    print("==== 🔄 Start to load model and tokenizer...")
    json_parser = PydanticOutputParser(pydantic_object=ParaphraseExtract, include_raw=True, strict=False)
    
    from constants import MODEL_PATHs

    model_name = MODEL_PATHs[args.model]
    print(f"Loading model from {model_name}...")
    
    system_rules = None
    if args.dataset == "myriadlama":
        from dataset import MyriadLamaDataset
        dataset = MyriadLamaDataset(model_name=args.model, debug=args.debug)
        system_rules = MYRIADLAMA_SYSTEM_RULES
    elif args.dataset == "commonsense":
        from dataset import CommonsenseParaphraseDataset
        dataset = CommonsenseParaphraseDataset(model_name=args.model, debug=args.debug)
        system_rules = COMMONSENSEQA_SYSTEM_RULES
    elif args.dataset == "mmlu":
        from dataset import MMLUParaphraseDataset
        dataset = MMLUParaphraseDataset(model_name=args.model, debug=args.debug)
        system_rules = MMLU_SYSTEM_RULES
    elif args.dataset == "logiqa":
        from dataset import LogiQAParaphraseDataset
        dataset = LogiQAParaphraseDataset(model_name=args.model, debug=args.debug)
    elif args.dataset == "hotpot":
        from dataset import HotpotDataset
        dataset = HotpotDataset(model_name=args.model, debug=args.debug)
        system_rules = HOTPOTQA_SYSTEM_RULES
    else:
        raise ValueError("Unsupported dataset. Please use 'webqa', 'myriadlama', 'commonsense', 'mmlu', 'logiqa', or 'hotpot'.")
    
    if system_rules is None:
        raise ValueError("System rules must be defined for the selected dataset.")
    
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)
        
    paraphrase_path = os.path.join(
        dataset.dataset_path, 
        f"paraphrases.num_para{args.num_paraphrase}.max_rounds{args.max_rounds}.temp{args.temperature}.topp{args.top_p}.jsonl")

    if os.path.exists(paraphrase_path) and not args.rewrite:
        paraphrases = {item["uuid"]: item for item in load_jsonl(paraphrase_path, verbose=True)}
    else:
        paraphrases = {}

    if len(paraphrases) == len(dataset.dataset) and not args.rewrite:
        print(f"All paraphrases already exist at {paraphrase_path}. Exiting.")
        exit(0)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map="auto", 
    ).eval()

    print(f"--> Starting paraphrase generation for dataset {args.dataset} using model {args.model}")
    print(f"--> Output paraphrase path: {paraphrase_path}")
    print(f"--> Current number of paraphrases loaded: {len(paraphrases)}")
    print()

    for idx, batch_data in tqdm(enumerate(dataloader), desc="Generating paraphrases", total=len(dataloader), dynamic_ncols=True):
        if not dataset.is_multi_choice:
            uuids, answers, all_paraphrases, _ = batch_data
        else:
            uuids, answers, all_paraphrases, _, _, _, _ = batch_data

        uuid = uuids[0]
        if uuid in paraphrases:
            if args.verbose:
                print(f"Skipping existing UUID: {uuid}")
            continue

        answer = answers[0]
        seed_prompt = all_paraphrases[0][0]
        raw_outputs, _paraphrases = enforced_paraphrase(
            system_rules,
            seed_prompt, 
            batch_size=args.batch_size, 
            temperature=args.temperature, top_p=args.top_p,
            num_paraphrase=args.num_paraphrase, max_rounds=args.max_rounds, 
            verbose=args.verbose)
        
        if args.verbose:
            print(f"-------- UUID: {uuid} --------")
            print("Seed prompt:", seed_prompt)
            print("Number of paraphrases:", len(_paraphrases))
            print("Paraphrases:", _paraphrases)
        
        paraphrases[uuid] = {
            "uuid": uuid,
            "answers": answer,
            "seed_prompt": seed_prompt,
            "auto_paraphrases": _paraphrases,
        }
        
    dump_jsonl(paraphrases.values(), paraphrase_path)
    print(f"===== ✅ Paraphrases saved to {paraphrase_path}\n\n")

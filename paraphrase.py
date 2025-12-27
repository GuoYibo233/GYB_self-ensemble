from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset
from utils import set_seed


def paraphrase_webqa_collate_fn(batch):
    questions = [item["question"] for item in batch]
    answers = [item["answers"] for item in batch]  # answers is a list of lists
    return questions, answers

def get_few_shot_paraphrases(few_shot=False, idx=0):
    instruction = """
Paraphrase the following question. Keep the original meaning, but use a different sentence structure and vocabulary. Aim to make the paraphrase sound natural and diverse.
    """

    example_prompts = [
        "who is the governor of hawaii now?",
        "what was nelson mandela's religion?",
        "who played sean in scrubs?",
        "what political party was henry clay?",
        "who are iran's major trading partners?"
    ]

    paraphrased_prompts = [
        [
            "as of now, who leads Hawaii as its governor?",
            "who's currently serving as Hawaii's governor?",
            "can you tell me who governs Hawaii right now?",
            "who’s in charge of the Hawaii state government these days?",
            "who’s the top executive official in Hawaii right now?"
        ],
        [
            "what was Mandela’s faith tradition",
            "can you tell me Mandela’s religion?",
            "what faith did Nelson Mandela practice?",
            "what was the religious affiliation of Nelson Mandela?",
            "what religion did Nelson Mandela follow?"
        ],
        [
            "which actor portrayed Sean in Scrubs?",
            "who took on the role of Sean in Scrubs?",
            "who played the character Sean in the TV show Scrubs?",
            "who was the actor that played Sean in the series Scrubs?",
            "do you know who played the part of Sean in Scrubs?"
        ],
        [
            "Henry Clay was a member of which political party?",
            "to which party did Henry Clay pledge his allegiance?",
            "what political affiliation did Henry Clay have?",
            "under which political banner did Henry Clay serve?",
            "where did Henry Clay stand on the political party map?"
        ],
        [
            "who does Iran trade with the most?",
            "who are the primary countries doing business with Iran?",
            "what are Iran’s strongest trade relationships?",
            "which countries top the list of Iran’s key trade allies?",
            "which countries are central to Iran’s import and export network?"
        ]
    ]

    few_shot_prompt = [f"Q: {prompt}\nParaphrase: {para}" for prompt, para in zip(example_prompts, [paras[idx] for paras in paraphrased_prompts])]
    if few_shot:
        prompts = f"{instruction}\n" + "\n\n".join(few_shot_prompt)
    else:
        prompts = f"{instruction}\n\n"
    return prompts


def generate_paraphrases(prompts, idx, seed=42):
    context = get_few_shot_paraphrases(few_shot=True, idx=idx)
    prompts = [f"{context}\n\nQ: {prompt}\nParaphrase:" for prompt in prompts]
    encoded = tokenizer(
        prompts, 
        padding=True, truncation=True,
        padding_side='left',
        return_tensors="pt",
        return_attention_mask=True)

    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    set_seed(seed)
    generated_ids = model.generate(
        input_ids, 
        max_new_tokens=30, 
        do_sample=True,
        temperature=1.5,
        attention_mask=attention_mask,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id)

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    new_generated_texts = [gen[len(prompt):].strip() for gen, prompt in zip(generated_texts, prompts)]
    return new_generated_texts

if __name__ == "__main__":
    import argparse
    import os

    from constants import MODEL_PATHs

    parser = argparse.ArgumentParser(description="Generate confidence scores for paraphrases.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run the model on.")
    parser.add_argument("--model", type=str, default="llama3.2_3b_it", help="Path to the pre-trained model.")
    parser.add_argument("--dataset", type=str, required=True, choices=["webqa", "myriadlama"], help="Dataset to use for generating paraphrases.")
    parser.add_argument("--rewrite", action="store_true", help="Rewrite the dataset if it already exists.")
    args = parser.parse_args()

    if args.dataset == "webqa":
        root = os.path.join("./webqa_test/", args.model)
    elif args.dataset == "myriadlama":
        root = os.path.join("./myriadlama")
    else:
        raise ValueError("Unsupported dataset. Please use 'webqa' or 'myriadlama'.")
    
    if not os.path.exists(root):
        os.makedirs(root)
    
    dump_path = os.path.join(root, "paraphrases_dataset")
    if os.path.exists(dump_path) and not args.rewrite:
        print(f"Confidence scores already exist at {dump_path}. Use --rewrite to overwrite.")
        exit(0)
    
    if args.dataset == "webqa":
        if args.model not in MODEL_PATHs:
            raise ValueError(f"Model {args.model} is not supported. Please choose from {list(MODEL_PATHs.keys())}.")
    
        model_path = MODEL_PATHs.get(args.model, args.model)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype="auto")
        tokenizer.pad_token = tokenizer.eos_token

        test_ds = load_dataset("stanfordnlp/web_questions", split="test")
        test_ds = test_ds.filter(lambda x: len(x["answers"]) == 1)
        dataloader = DataLoader(test_ds, batch_size=8, collate_fn=paraphrase_webqa_collate_fn)

        paras = []
        for i in range(5):
            print(f"Generating paraphrases for iteration {i+1}")
            all_paraphrases = []
            for questions, answers in tqdm(dataloader):
                answers = [ans[0] for ans in answers]
                generations = generate_paraphrases(questions, idx=i, seed=i)
                paraphrases = [gen.strip().split('\n')[0] for gen in generations]
                all_paraphrases.extend(paraphrases)
            paras.append(all_paraphrases)

        ds = dataloader.dataset.add_column("paraphrase1", paras[0])
        ds = ds.add_column("paraphrase2", paras[1])
        ds = ds.add_column("paraphrase3", paras[2])
        ds = ds.add_column("paraphrase4", paras[3])
        ds = ds.add_column("paraphrase5", paras[4])
        ds.save_to_disk(dump_path)
    elif args.dataset == "myriadlama":
        ds = load_dataset("iszhaoxin/MyriadLAMA", split="train")
        df = ds.to_pandas()

        items = []
        for uuid, sdf in tqdm(df.groupby('uuid')):
            uuid = sdf['uuid'].iloc[0]
            rel = sdf['rel_uri'].iloc[0]
            answers = sdf['obj_aliases'].iloc[0].tolist()
            manual_templates = sdf[sdf['is_manual']==True]['template'].tolist()
            manual_prompts = [template.replace('[X]', sdf['sub_ent'].iloc[0]).replace('[Y]', '[MASK]') for template in manual_templates]
            auto_templates = sdf[sdf['is_manual']==False]['template'].tolist()
            auto_prompts = [template.replace('[X]', sdf['sub_ent'].iloc[0]).replace('[Y]', '[MASK]') for template in auto_templates]
            items.append({
                "uuid": uuid,
                "rel": rel,
                "answers": answers,
                "manual_paraphrases": manual_prompts,
                "auto_paraphrases": auto_prompts
            })
        
        import pandas as pd

        from datasets import Dataset
        newdf = pd.DataFrame(items)
        newdf = newdf.sample(1000, random_state=42)
        ds = Dataset.from_pandas(newdf)
        ds.save_to_disk(dump_path)


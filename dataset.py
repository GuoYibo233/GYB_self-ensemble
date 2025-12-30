import ast
import hashlib
import os
import random
from abc import abstractmethod
from pdb import set_trace

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import MODEL_PATHs
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from utils import DATASET_ROOT, PROJECT_DATASET_ROOT, set_seed

COMMONSENSE_PARAPHRASE_PATH = "/home/y-guo/self-ensemble/new_datasets/my_commonsense_paraphrase"
MMLA_PARAPHRASE_PATH = "/home/y-guo/self-ensemble/new_datasets/my_mmlu_paraphrase"
LOGIQA_PARAPHRASE_PATH = "/home/y-guo/self-ensemble/new_datasets/my_logiqa_paraphrase"
HOTPOT_PARAPHRASE_PATH = "/home/y-guo/self-ensemble/new_datasets/my_hotpot_paraphrase"

def string_to_id(s):
    return hashlib.md5(s.encode()).hexdigest()

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


def generate_paraphrases(model, tokenizer, prompts, idx, seed=42):
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

def webqa_collate_fn(batch):
    questions = [item["question"] for item in batch]
    answers = [item["answers"] for item in batch]  # answers is a list of lists
    return questions, answers

class ParaPharaseDataset:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

        if not os.path.exists(self.dataset_root):
            os.makedirs(self.dataset_root, exist_ok=True)

        self.ds = self.load_dataset()

    @property
    def dataset_root(self):
        pass

    @property
    def dataset_path(self):
        pass

    @property
    def instruction(self):
        pass

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def get_dataloader(self, batch_size=8, shuffle=False):
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass

    @abstractmethod
    def get_few_shot_examples(self, k=5, seed=42):
        pass

    def format_example(self, example):
        question = example["question"]
        answer = example["answers"][0]
        return f"Q: {question}\nA: {answer}"

    def construct_prompts(self, few_shot_examples, questions, instruction=None):
        if instruction is None:
            instruction = self.instruction
        prompts = [f"{instruction}\n\n{few_shot_examples}\n\nQ: {question}\nA:" for question in questions]
        return prompts

    def construct_prompts_with_paraphrases(self, few_shot_examples, paraphrases):
        context = f"{self.instruction}\n\n{few_shot_examples}\n\n" if few_shot_examples else f"{self.instruction}\n\n"
        paraphrase_qs = [f"Q: {question}\nA:" for question in paraphrases]
        paraphrases = "".join(paraphrase_qs)
        metadata = {
            "len_context": len(context),
            "len_paras": [len(question) for question in paraphrase_qs],
        }
        return f"{context}{paraphrases}", metadata

    def construct_prompts_single_para_qapair(self, few_shot_examples, paraphrases):
        context = f"{self.instruction}\n\n{few_shot_examples}\n\nQ: " if few_shot_examples else f"{self.instruction}\n\nQ: "
        paraphrase_qs = [f"{question}\n" for question in paraphrases]
        paraphrases = "".join(paraphrase_qs)
        answer = "A:"
        metadata = {
            "len_context": len(context),
            "len_paras": [len(question) for question in paraphrase_qs],
            "len_answer": len(answer),
        }
        return f"{context}{paraphrases}{answer}", metadata

    def construct_explicit_prompts(self, paraphrases):
        context = f"{self.instruction}\n"
        paraphrase_qs = [f"{question}\n" for question in paraphrases]
        paraphrases = "".join(paraphrase_qs)
        metadata = {
            "len_context": len(context),
            "len_paras": [len(question) for question in paraphrase_qs],
        }
        return f"{context}{paraphrases}", metadata

class WebQADataset(ParaPharaseDataset):
    def __init__(self, model_name, device="auto"):
        self.model_name = model_name
        self.device = device
        self.train_ds = None
        super().__init__("webqa", model_name)

    @property
    def dataset_root(self):
        return os.path.join(DATASET_ROOT, "webqa", self.model_name)

    @property
    def dataset_path(self):
        return os.path.join(self.dataset_root, "paraphrases_dataset")

    @property
    def instruction(self):
        return "Answer the question based on general world knowledge. Provide a short and direct answer."

    def load_dataset(self):
        if os.path.exists(self.dataset_path):
            print(f"Dataset already exists at {self.dataset_path}. Loading from disk.")
            return load_from_disk(self.dataset_path)

        print("Creating WebQA dataset...")
        if self.model_name not in MODEL_PATHs:
            raise ValueError(f"Model {self.model_name} is not supported. Please choose from {list(MODEL_PATHs.keys())}.")

        model_path = MODEL_PATHs.get(self.model_name)
        print(f"Loading model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype="auto")
        tokenizer.pad_token = tokenizer.eos_token

        test_ds = load_dataset("stanfordnlp/web_questions", split="test")
        dataloader = DataLoader(test_ds, batch_size=8, collate_fn=webqa_collate_fn)

        paras = []
        for i in range(5):
            print(f"Generating paraphrases for iteration {i+1}")
            all_paraphrases = []
            all_questions = []
            for questions, answers in tqdm(dataloader, desc="Generating paraphrases", dynamic_ncols=True):
                answers = [ans[0] for ans in answers]
                generations = generate_paraphrases(model, tokenizer, questions, idx=i, seed=i)
                paraphrases = [gen.strip().split('\n')[0] for gen in generations]
                all_questions.extend(questions)
                all_paraphrases.extend(paraphrases)
            paras.append(all_paraphrases)

        ds = dataloader.dataset.add_column("uuid", [string_to_id(question) for question in all_questions])
        ds = ds.add_column("paraphrase0", all_questions)
        ds = ds.add_column("paraphrase1", paras[0])
        ds = ds.add_column("paraphrase2", paras[1])
        ds = ds.add_column("paraphrase3", paras[2])
        ds = ds.add_column("paraphrase4", paras[3])
        ds = ds.add_column("paraphrase5", paras[4])
        ds.save_to_disk(self.dataset_path)
        return ds

    def get_dataloader(self, batch_size=8, shuffle=False):
        return DataLoader(self.ds, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle)

    def collate_fn(self, batch):
        uuids = [item["uuid"] for item in batch]
        prompt0 = [item["paraphrase0"] for item in batch]
        prompt1 = [item["paraphrase1"] for item in batch]
        prompt2 = [item["paraphrase2"] for item in batch]
        prompt3 = [item["paraphrase3"] for item in batch]
        prompt4 = [item["paraphrase4"] for item in batch]
        prompt5 = [item["paraphrase5"] for item in batch]
        answers = [item["answers"] for item in batch]
        all_prompts = [prompt0, prompt1, prompt2, prompt3, prompt4, prompt5]
        return uuids, answers, all_prompts

    def get_few_shot_examples(self, k=5, seed=42):
        if self.train_ds is None:
            self.train_ds = load_dataset("stanfordnlp/web_questions", split="train")
        random.seed(seed)
        indices = random.sample(range(len(self.train_ds)), k)
        return "\n\n".join(self.format_example(self.train_ds[i]) for i in indices)

class MyriadLamaDataset(ParaPharaseDataset):

    def __init__(self, model_name, debug=False):
        self.model_name = model_name
        self.debug = debug
        if self.debug:
            print("Debug mode: using a smaller subset of the dataset.")
            super().__init__("myriadlama-debug", model_name)
        else:
            super().__init__("myriadlama", model_name)

    @property
    def dataset_root(self):
        if self.debug:
            return os.path.join(
                PROJECT_DATASET_ROOT, "myriadlama-debug", self.model_name
            )
        else:
            return os.path.join(PROJECT_DATASET_ROOT, "myriadlama", self.model_name)

    @property
    def dataset_path(self):
        return os.path.join(self.dataset_root, "paraphrases_dataset")

    @property
    def instruction(self):
        return "Based on the context, predict the [MASK] in the sentence in one word."
        # return "Predict the [MASK] in the sentence in one word."

    def load_dataset(self):
        if os.path.exists(self.dataset_path):
            print(f"Dataset already exists at {self.dataset_path}. Loading from disk.")
            return load_from_disk(self.dataset_path)['test']

        print("Creating MyriadLAMA dataset...")
        ds = load_dataset("iszhaoxin/MyriadLAMA", split="train")
        df = ds.to_pandas()

        items = []
        for uuid, sdf in tqdm(df.groupby('uuid'), desc="Processing MyriadLAMA dataset", dynamic_ncols=True):
            uuid = sdf['uuid'].iloc[0]
            rel = sdf['rel_uri'].iloc[0]
            answers = sdf['obj_aliases'].iloc[0].tolist()
            manual_templates = sdf[sdf['is_manual'] == True]['template'].tolist()
            manual_prompts = [template.replace('[X]', sdf['sub_ent'].iloc[0]).replace('[Y]', '[MASK]') for template in manual_templates]
            auto_templates = sdf[sdf['is_manual'] == False]['template'].tolist()
            auto_prompts = [template.replace('[X]', sdf['sub_ent'].iloc[0]).replace('[Y]', '[MASK]') for template in auto_templates]
            items.append({
                "uuid": uuid,
                "rel": rel,
                "answers": answers,
                "manual_paraphrases": manual_prompts,
                "auto_paraphrases": auto_prompts
            })

        newdf = pd.DataFrame(items)
        ds = Dataset.from_pandas(newdf)
        if self.debug:
            ds = ds.train_test_split(test_size=200, seed=42, shuffle=True)
        else:
            ds = ds.train_test_split(test_size=2000, seed=42, shuffle=True)
        ds.save_to_disk(self.dataset_path)
        return ds['test']

    def get_dataloader(self, batch_size=8, shuffle=False):
        return DataLoader(self.ds, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle)

    def collate_fn(self, batch):
        uuids = [item["uuid"] for item in batch]
        answers = [item["answers"] for item in batch]
        paraphrases = []
        for item in batch:
            uuid = item["uuid"]
            random.seed(uuid)
            manual_list = item["manual_paraphrases"]
            auto_list = item["auto_paraphrases"]

            # Select exactly 5 manual + 5 auto paraphrases per item when available
            if len(manual_list) < 5:
                print(f"⚠️ MyriadLAMA uuid {uuid}: manual paraphrase count {len(manual_list)} < 5")
            if len(auto_list) < 5:
                print(f"⚠️ MyriadLAMA uuid {uuid}: auto paraphrase count {len(auto_list)} < 5")

            manual_sel = (
                random.sample(manual_list, 5)
                if len(manual_list) >= 5 else manual_list[:]
            )
            auto_sel = (
                random.sample(auto_list, 5)
                if len(auto_list) >= 5 else auto_list[:]
            )

            merged = manual_sel + auto_sel
            if len(merged) < 10:
                print(f"⚠️ MyriadLAMA uuid {uuid}: total paraphrases {len(merged)} < 10 (after selection)")
            paraphrases.append(merged)
        return uuids, answers, list(zip(*paraphrases))

    def get_few_shot_examples(self, k=5, seed=42):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}. Please run the dataset preparation first.")

        train_ds = load_from_disk(self.dataset_path)['train']
        random.seed(seed)
        indices = random.sample(range(len(train_ds)), k)
        return "\n\n".join(self.format_example(train_ds[i]) for i in indices)

    def format_example(self, example):
        question = example["manual_paraphrases"][0]
        answer = example["answers"][0]
        return f"Q: {question}\nA: {answer}"


class MultiChoiceParaphraseDataset(ParaPharaseDataset):
    """
    Base class for multi-choice QA paraphrase datasets.
    Aggregates paraphrase variants per orig_id.
    Each aggregated row contains:
      - uuid: orig_id
      - paraphrases: list[str] sorted by paraphrase_idx
      - answers: [correct_choice_text] (first element is gold)
      - choices_label / choices_text / answer_label kept for reference
    """

    def __init__(self, model_name, raw_path: str, dataset_type: str = "commonsense", debug=False):
        self.model_name = model_name
        self.raw_dataset_path = raw_path
        self.dataset_type = dataset_type
        self.debug = debug
        if self.debug:
            super().__init__(f"{dataset_type}_paraphrase", model_name)
        else:
            super().__init__(f"{dataset_type}_paraphrase-debug", model_name)

    @property
    def dataset_root(self):
        if self.debug:
            return os.path.join(PROJECT_DATASET_ROOT, f"{self.dataset_type}_paraphrase-debug", self.model_name)
        return os.path.join(PROJECT_DATASET_ROOT, f"{self.dataset_type}_paraphrase", self.model_name)

    @property
    def dataset_path(self):
        return os.path.join(self.dataset_root, "paraphrases_dataset")

#     @property
#     def instruction(self):
#         return """Answer the following multiple-choice question by selecting the correct option (A, B, C, D, or E).
# Output exactly one capital letter corresponding to the chosen option. Do not output punctuation, text, or explanations"""

    @property
    def instruction(self):
        return """You are given a multiple-choice question.
Choose the correct answer from {A, B, C, D, E}.
Return ONLY one capital letter from {A, B, C, D, E}.
Do NOT output anything else.

Format of input and output:
Question:
{question}

Options:
A. {option A}
B. {option B}
C. {option C}
D. {option D}
E. {option E}

Answer = <one letter>
"""

    def construct_multi_choice_prompts(self, few_shot_examples, paraphrases, choices_labels, choices_texts):
        options_str = "\n".join([f"{label}. {text}" for label, text in zip(choices_labels, choices_texts)])
        
        prompts = []
        for paraphrase in paraphrases:
            if few_shot_examples:
                prompt = f"{self.instruction}\n\n{few_shot_examples}\n\nQuestion:\n{paraphrase}\n\nOptions:\n{options_str}\n\nAnswer = "
            else:
                prompt = f"{self.instruction}\n\nQuestion:\n{paraphrase}\n\nOptions:\n{options_str}\n\nAnswer = "
            prompts.append(prompt)
        return prompts
    

    def load_dataset(self):
        if os.path.exists(self.dataset_path):
            print(f"Dataset already exists at {self.dataset_path}. Loading from disk.")
            return load_from_disk(self.dataset_path)

        print(f"Loading raw {self.dataset_type} paraphrase dataset from {self.raw_dataset_path}")
        raw_ds = load_from_disk(self.raw_dataset_path)
        if isinstance(raw_ds, DatasetDict):
            raw_ds = raw_ds["train"]
        df = raw_ds.to_pandas() 
        
        items = []
        for cnt, (orig_id, sdf) in tqdm(enumerate(df.groupby("orig_id")), desc=f"Processing {self.dataset_type} paraphrases", dynamic_ncols=True):
            sdf = sdf.sort_values("paraphrase_idx")
            paraphrases = sdf["question"].tolist()
            first = sdf.iloc[0]
            labels = first["choices"]["label"]
            texts = first["choices"]["text"]
            label2text = {l: t for l, t in zip(labels, texts)}
            answer_key = first["answerKey"]
            answer_text = label2text.get(answer_key, "")
            items.append(
                {
                    "uuid": orig_id,
                    "paraphrases": paraphrases[:10],
                    "answers": [answer_text],
                    "answer_label": answer_key,
                    "choices_label": labels,
                    "choices_text": texts,
                    "orig_question": first.get("orig_question", ""),
                    "question_concept": first.get("question_concept", ""),
                }
            )

            if len(paraphrases) < 10:
                print(f"⚠️ {self.dataset_type} uuid {orig_id}: paraphrase count {len(paraphrases)} < 10")

            if self.debug and cnt >= 100:
                break

        agg_ds = Dataset.from_pandas(pd.DataFrame(items))
        agg_ds.save_to_disk(self.dataset_path)
        return agg_ds

    def get_dataloader(self, batch_size=8, shuffle=False):
        return DataLoader(self.ds, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle)

    def collate_fn(self, batch):
        uuids = [item["uuid"] for item in batch]
        # answers = [item["answers"] for item in batch]
        paraphrases = [item["paraphrases"] for item in batch]
        choices_labels = [item["choices_label"] for item in batch]
        choices_texts = [item["choices_text"] for item in batch]
        answer_labels = [item["answer_label"] for item in batch]
        return uuids, answer_labels, list(zip(*paraphrases)), choices_labels, choices_texts, answer_labels

    def get_few_shot_examples(self, k=5, seed=42, is_ppl_format=False):
        random.seed(seed)
        indices = random.sample(range(len(self.ds)), k)
        return "\n\n".join(self.format_example(self.ds[i], is_ppl_format=is_ppl_format) for i in indices)

    def format_example(self, example, is_ppl_format=False):
        question = example["paraphrases"][0]
        answer_label = example["answer_label"]
        choices_label = example["choices_label"]
        choices_text = example["choices_text"]
        
        if is_ppl_format:
            return f"Q: {question}\nA: {choices_text[choices_label.index(answer_label)]}"
        else:
            options_str = "\n".join([f"{label}. {text}" for label, text in zip(choices_label, choices_text)])
            return f"Question:\n{question}\n\nOptions:\n{options_str}\n\nAnswer = {answer_label}"


class CommonsenseParaphraseDataset(MultiChoiceParaphraseDataset):
    """Commonsense QA paraphrase dataset."""
    def __init__(self, model_name, raw_path: str = COMMONSENSE_PARAPHRASE_PATH, debug=False):
        super().__init__(model_name, raw_path, dataset_type="commonsense", debug=debug)


class MMLUParaphraseDataset(MultiChoiceParaphraseDataset):
    """MMLU (Massive Multitask Language Understanding) paraphrase dataset."""
    def __init__(self, model_name, raw_path: str = MMLA_PARAPHRASE_PATH, debug=False):
        super().__init__(model_name, raw_path, dataset_type="mmlu", debug=debug)


class LogiQAParaphraseDataset(MultiChoiceParaphraseDataset):
    """LogiQA paraphrase dataset."""
    def __init__(self, model_name, raw_path: str = LOGIQA_PARAPHRASE_PATH, debug=False):
        super().__init__(model_name, raw_path, dataset_type="logiqa", debug=debug)


class HotpotDataset(ParaPharaseDataset):
    """HotpotQA paraphrase dataset: 1 manual + 10 auto paraphrases per uuid."""

    def __init__(self, model_name, debug=False):
        self.model_name = model_name
        self.debug = debug
        if self.debug:
            print("Debug mode: using a smaller subset of the dataset.")
            super().__init__("hotpot-debug", model_name)
        else:
            super().__init__("hotpot", model_name)

    @property
    def dataset_root(self):
        if self.debug:
            return os.path.join(
                PROJECT_DATASET_ROOT, "hotpot-debug", self.model_name
            )
        else:
            return os.path.join(PROJECT_DATASET_ROOT, "hotpot", self.model_name)

    @property
    def dataset_path(self):
        return os.path.join(self.dataset_root, "paraphrases_dataset")

    @property
    def instruction(self):
        return "Answer the question based on the provided context in one or two sentences."

    def load_dataset(self):
        if os.path.exists(self.dataset_path):
            print(f"Dataset already exists at {self.dataset_path}. Loading from disk.")
            return load_from_disk(self.dataset_path)

        print("Loading HotpotQA dataset...")
        ds = load_from_disk(HOTPOT_PARAPHRASE_PATH)
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
        
        print(f"Dataset loaded with {len(ds)} items")

        items = []
        skipped_count = 0
        
        for idx in tqdm(range(len(ds)), desc="Processing HotpotQA dataset", dynamic_ncols=True):
            item = ds[idx]
            uuid = item['uuid']
            
            # Parse answer field - HotpotQA uses 'answer' (singular) not 'answers'
            raw_answer = item['answer']
            if isinstance(raw_answer, str):
                answers = [raw_answer]  # Convert single answer to list
            elif isinstance(raw_answer, list):
                answers = raw_answer
            else:
                answers = [str(raw_answer)]
            
            # Parse question as manual_paraphrases (use original question)
            raw_question = item['question']
            if isinstance(raw_question, str):
                manual_paraphrases = [raw_question]  # Original question is the manual paraphrase
            elif isinstance(raw_question, list):
                manual_paraphrases = raw_question[:1]  # Take first if it's a list
            else:
                manual_paraphrases = [str(raw_question)]
            
            # Parse auto_paraphrases - handle both list and string representations
            raw_auto = item['auto_paraphrases']
            if isinstance(raw_auto, str):
                try:
                    auto_paraphrases = ast.literal_eval(raw_auto)
                    if not isinstance(auto_paraphrases, list):
                        auto_paraphrases = [auto_paraphrases]
                except Exception as e:
                    if skipped_count < 5:
                        print(f"⚠️ Failed to parse auto_paraphrases for uuid {uuid}: {e}")
                    auto_paraphrases = [raw_auto]
            elif isinstance(raw_auto, list):
                auto_paraphrases = raw_auto
            else:
                auto_paraphrases = []
            
            # Check if paraphrase counts meet requirements
            if len(manual_paraphrases) < 1 or len(auto_paraphrases) < 10:
                if skipped_count < 5:  # Only print first 5 to avoid spam
                    print(f"⚠️ Skipping uuid {uuid}: manual={len(manual_paraphrases)}, auto={len(auto_paraphrases)}")
                skipped_count += 1
                continue
            
            items.append({
                "uuid": uuid,
                "answers": answers,
                "manual_paraphrases": manual_paraphrases,
                "auto_paraphrases": auto_paraphrases
            })

            if self.debug and idx >= 100:
                break

        print(f"✓ Processed {len(items)} items, skipped {skipped_count} items with insufficient paraphrases")
        
        if len(items) == 0:
            raise ValueError("No valid items found in dataset. All items were filtered out.")

        newdf = pd.DataFrame(items)
        ds = Dataset.from_pandas(newdf)
        ds.save_to_disk(self.dataset_path)
        return ds

    def get_dataloader(self, batch_size=8, shuffle=False):
        return DataLoader(self.ds, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle)

    def collate_fn(self, batch):
        uuids = [item["uuid"] for item in batch]
        answers = [item["answers"] for item in batch]
        paraphrases = []
        for item in batch:
            uuid = item["uuid"]
            random.seed(uuid)
            manual_list = item["manual_paraphrases"]
            auto_list = item["auto_paraphrases"]

            # For HotpotQA: use 1 manual + 10 auto paraphrases per item
            # Data already filtered during load_dataset, so we can assume valid counts
            manual_sel = manual_list[:1]
            auto_sel = random.sample(auto_list, 10)

            merged = manual_sel + auto_sel

            paraphrases.append(merged)
        return uuids, answers, list(zip(*paraphrases))

    def get_few_shot_examples(self, k=5, seed=42):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}. Please run the dataset preparation first.")

        full_ds = load_from_disk(self.dataset_path)
        random.seed(seed)
        indices = random.sample(range(len(full_ds)), k)
        return "\n\n".join(self.format_example(full_ds[i]) for i in indices)

    def format_example(self, example):
        question = example["manual_paraphrases"][0] if isinstance(example["manual_paraphrases"], list) else example["manual_paraphrases"]
        answer = example["answers"][0] if isinstance(example["answers"], list) else example["answers"]
        return f"Q: {question}\nA: {answer}"
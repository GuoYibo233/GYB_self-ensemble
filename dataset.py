import hashlib
import os
import random
from abc import abstractmethod

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import MODEL_PATHs
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from utils import DATASET_ROOT, PROJECT_DATASET_ROOT, load_jsonl, set_seed

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
    def __init__(self, dataset, model, paraphrase_file: str = None):
        self.dataset = dataset
        self.model = model
        
        self.num_additional_paraphrases = None
        self.additional_paraphrases = None
        paraphrase_file = os.path.join(self.dataset_path, paraphrase_file) if paraphrase_file is not None else None
        if paraphrase_file is not None:
            assert os.path.exists(paraphrase_file), f"Paraphrase file {paraphrase_file} does not exist."
            self.additional_paraphrases = {item["uuid"]: item for item in load_jsonl(paraphrase_file)}
            self.num_additional_paraphrases = max(len(item["auto_paraphrases"]) for item in self.additional_paraphrases.values()) + 1
            self.num_additional_paraphrases += 1

        if not os.path.exists(self.dataset_root):
            os.makedirs(self.dataset_root, exist_ok=True)
        self.ds = self.load_dataset()

    @property
    def dataset_path(self):
        pass

    @property
    def instruction(self):
        pass

    @property
    def choice_labels(self):
        return None
        
    @property
    def is_multi_choice(self):
        return False

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

    def construct_prompts_for_reasoning(self, few_shot_examples, questions):
        assert few_shot_examples == "", f"Few-shot examples are not supported for reasoning generation. but got {few_shot_examples}."
        prompts = questions
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
        self.dataset_root = os.path.join(DATASET_ROOT, "webqa", self.model_name)
        super().__init__("webqa", model_name)

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

    def __init__(self, model_name, debug=False, paraphrase_file: str = None):
        self.model_name = model_name
        self.debug = debug
        dataset_name = "myriadlama-debug" if self.debug else "myriadlama"
        super().__init__(dataset_name, model_name, paraphrase_file)
        self.dataset_name = os.path.join(PROJECT_DATASET_ROOT, dataset_name, self.model_name)
    
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

    def get_dataloader(self, batch_size=1, shuffle=False):
        return DataLoader(self.ds, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle)

    def collate_fn(self, batch):
        uuids = [item["uuid"] for item in batch]
        answers = [item["answers"] for item in batch]
        paraphrases = []
        is_origs = []
        for item in batch:
            uuid = item["uuid"]
            random.seed(uuid)

            if self.additional_paraphrases is not None:
                assert uuid in self.additional_paraphrases, f"⚠️ MyriadLAMA uuid {uuid} not found in additional paraphrases file"
                _paraphrases = [self.additional_paraphrases[uuid]["seed_prompt"]] + self.additional_paraphrases[uuid]["auto_paraphrases"]
                is_origs.append([True] + [False]*len(self.additional_paraphrases[uuid]["auto_paraphrases"]))
            else:
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

                _paraphrases = manual_sel + auto_sel
                if len(_paraphrases) < 10:
                    print(f"⚠️ MyriadLAMA uuid {uuid}: total paraphrases {len(_paraphrases)} < 10 (after selection)")
                is_origs.append([True]*5 + [False]*5)
            paraphrases.append(_paraphrases)
        return uuids, answers, list(zip(*paraphrases)), list(zip(*is_origs))

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

class MyriadLama100Dataset(MyriadLamaDataset):
    def __init__(self, model_name, debug=False, paraphrase_file: str = None):
        super().__init__(model_name, debug, paraphrase_file)
        dataset_name = "myriadlama100-debug" if self.debug else "myriadlama100"
        self.dataset_root = os.path.join(PROJECT_DATASET_ROOT, dataset_name, self.model_name)
    
    def collate_fn(self, batch):
        uuids = [item["uuid"] for item in batch]
        answers = [item["answers"] for item in batch]
        paraphrases = []
        is_origs = []
        for item in batch:
            uuid = item["uuid"]
            random.seed(uuid)
            manual_list = item["manual_paraphrases"]
            auto_list = item["auto_paraphrases"]
            merged = manual_list + auto_list
            assert len(merged) == 100, f"⚠️ MyriadLAMA uuid {uuid}: paraphrase merging error"
            paraphrases.append(merged)
            is_origs.append([True]*5 + [False]*95)

        return uuids, answers, list(zip(*paraphrases)), list(zip(*is_origs))

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

    def __init__(
            self, model_name, raw_path: str, 
            dataset_type: str = "commonsense", debug=False, 
            paraphrase_file: str = None):
        self.model_name = model_name
        self.raw_dataset_path = raw_path
        self.dataset_type = dataset_type
        self.debug = debug
        dataset_name = f"{dataset_type}-debug" if self.debug else f"{dataset_type}"
        self.dataset_root = os.path.join(PROJECT_DATASET_ROOT, dataset_name, self.model_name)
        super().__init__(f"{dataset_type}", model_name, paraphrase_file)
        
    @property
    def dataset_path(self):
        return os.path.join(self.dataset_root, "paraphrases_dataset")

    @property
    def is_multi_choice(self):
        return True

    @property
    def instruction(self):
        return getattr(
            self, 
            "_instruction", 
            """You are given a multiple-choice question.
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
""")

    @instruction.setter
    def instruction(self, instruction):
        self._instruction = instruction
    
    def construct_prompts_with_paraphrases(self, few_shot_examples, paraphrases):
        context = f"{self.instruction}\n\n{few_shot_examples}\n\n" if few_shot_examples else f"{self.instruction}\n\n"
        paraphrase_qs = [f"Q: {question}\nA:" for question in paraphrases]
        paraphrases = "".join(paraphrase_qs)
        metadata = {
            "len_context": len(context),
            "len_paras": [len(question) for question in paraphrase_qs],
        }
        return f"{context}{paraphrases}", metadata

    def construct_multi_choice_prompts(self, few_shot_examples, paraphrases, choices_labels, choices_texts):
        options_str = "\n".join([f"{label}. {text}" for label, text in zip(choices_labels, choices_texts)])
        
        prompts = []
        answer_part = "Answer =" if self.choice_labels[0][0] == " " else "Answer = "
        for paraphrase in paraphrases:
            if few_shot_examples:
                prompt = f"{self.instruction}\n\n{few_shot_examples}\n\nQuestion:\n{paraphrase}\n\nOptions:\n{options_str}\n\n{answer_part}"
            else:
                prompt = f"{self.instruction}\n\nQuestion:\n{paraphrase}\n\nOptions:\n{options_str}\n\n{answer_part}"
            prompts.append(prompt)
        return prompts
    
    def construct_prompts_single_para_qapair(self, few_shot_examples, paraphrases, choices_labels, choices_texts):
        if few_shot_examples:
            context_str = f"{self.instruction}\n\n{few_shot_examples}\n\nQuestion:\n"
        else:
            context_str = f"{self.instruction}\n\nQuestion:\n"

        paraphrase_qs = [f"{question}\n" for question in paraphrases]
        paraphrase_str = "".join(paraphrase_qs)
        
        options_str = "\n".join([f"{label}. {text}" for label, text in zip(choices_labels, choices_texts)])
        answer_part = "Answer =" if self.choice_labels[0][0] == " " else "Answer = "
        answer_str = f"\nOptions:\n{options_str}\n\n{answer_part}"
    
        metadata = {
            "len_context": len(context_str),
            "len_paras": [len(question) for question in paraphrase_qs],
            "len_answer": len(answer_str),
        }
        return f"{context_str}{paraphrase_str}{answer_str}", metadata
    
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
            first = sdf.iloc[0]
            labels = first["choices"]["label"]
            texts = first["choices"]["text"]
            label2text = {label: text for label, text in zip(labels, texts)}
            answer_key = first["answerKey"]
            answer_text = label2text.get(answer_key, "")
            orig_question = first.get("orig_question", "")
            
            if self.additional_paraphrases is None:
                _paraphrases = [orig_question] + sdf["question"].tolist()[:4]
                _is_origs = [True] + [False] * 4
            else:
                assert orig_id in self.additional_paraphrases, f"⚠️ {self.dataset_type} uuid {orig_id} not found in additional paraphrases file"
                _paraphrases = [self.additional_paraphrases[orig_id]["seed_prompt"]] + self.additional_paraphrases[orig_id]["auto_paraphrases"]
                _is_origs = [True] + [False] * len(self.additional_paraphrases[orig_id]["auto_paraphrases"])

            items.append(
                {
                    "uuid": orig_id,
                    "paraphrases": _paraphrases,
                    "is_orig": _is_origs,
                    "answers": [answer_text],
                    "answer_label": answer_key,
                    "choices_label": labels,
                    "choices_text": texts,
                    "orig_question": orig_question,
                    "question_concept": first.get("question_concept", ""),
                }
            )
            if self.debug and cnt >= 199:
                break

        agg_ds = Dataset.from_pandas(pd.DataFrame(items))
        agg_ds.save_to_disk(self.dataset_path)
        return agg_ds

    def get_dataloader(self, batch_size=8, shuffle=False):
        return DataLoader(self.ds, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle)

    def collate_fn(self, batch):
        uuids, choices_labels, choices_texts, answer_labels, is_origs, paraphrases = [], [], [], [], [], []
        # uuids = [item["uuid"] for item in batch]
        # choices_labels = [item["choices_label"] for item in batch]
        # choices_texts = [item["choices_text"] for item in batch]
        # answer_labels = [item["answer_label"] for item in batch]
        # is_origs = [item["is_orig"] for item in batch]
        # paraphrases = []
        for item in batch:
            uuids.append(item["uuid"])
            choices_labels.append(item["choices_label"])
            choices_texts.append(item["choices_text"])
            answer_labels.append(item["answer_label"])
            if self.additional_paraphrases is None:
                assert len(item["paraphrases"]) == 5, f"⚠️ {self.dataset_type} uuid {item['uuid']}: paraphrase count {len(item['paraphrases'])} != 5"
                paraphrases.append(item["paraphrases"])
                is_origs.append(item["is_orig"])
            else:
                _paraphrases = [self.additional_paraphrases[item["uuid"]]["seed_prompt"]] + self.additional_paraphrases[item["uuid"]]["auto_paraphrases"]
                paraphrases.append(_paraphrases)
                is_origs.append([True] + [False]*len(self.additional_paraphrases[item["uuid"]]["auto_paraphrases"]))
        return uuids, answer_labels, list(zip(*paraphrases)), choices_labels, choices_texts, answer_labels, list((zip(*is_origs)))

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
    def __init__(self, model_name, raw_path: str = COMMONSENSE_PARAPHRASE_PATH, debug=False, paraphrase_file: str = None):
        super().__init__(model_name, raw_path, dataset_type="commonsense", debug=debug, paraphrase_file=paraphrase_file)

    @property
    def choice_labels(self):
        return getattr(self, "_choice_labels", [' A', ' B', ' C', ' D', ' E'])

    @choice_labels.setter
    def choice_labels(self, labels):
        self._choice_labels = labels

class MMLUParaphraseDataset(MultiChoiceParaphraseDataset):
    """MMLU (Massive Multitask Language Understanding) paraphrase dataset."""
    def __init__(self, model_name, raw_path: str = MMLA_PARAPHRASE_PATH, debug=False, paraphrase_file: str = None):
        super().__init__(model_name, raw_path, dataset_type="mmlu", debug=debug, paraphrase_file=paraphrase_file)

    @property
    def choice_labels(self):
        return getattr(self, "_choice_labels", [' A', ' B', ' C', ' D'])

    @choice_labels.setter
    def choice_labels(self, labels):
        self._choice_labels = labels

class LogiQAParaphraseDataset(MultiChoiceParaphraseDataset):
    """LogiQA paraphrase dataset."""
    def __init__(self, model_name, raw_path: str = LOGIQA_PARAPHRASE_PATH, debug=False, paraphrase_file: str = None):
        super().__init__(model_name, raw_path, dataset_type="logiqa", debug=debug, paraphrase_file=paraphrase_file)

    @property
    def choice_labels(self):
        return getattr(self, "_choice_labels", [' A', ' B', ' C', ' D'])
    
    @choice_labels.setter
    def choice_labels(self, labels):
        self._choice_labels = labels


class HotpotDataset(ParaPharaseDataset):
    """HotpotQA paraphrase dataset: 1 manual + 10 auto paraphrases per uuid."""

    def __init__(
            self, 
            model_name, 
            debug=False, 
            reasoning=False,
            num_paraphrases: int = None,
            paraphrase_file: str = None
        ):

        self.model_name = model_name
        self.debug = debug
        self.reasoning = reasoning
        self.num_paraphrases = 4 if num_paraphrases is None else num_paraphrases
        suffix = "-reasoning" if self.reasoning else ""
        if self.debug:
            suffix += "-debug"
        
        self.dataset_name = "hotpot" + suffix
        self.dataset_root = os.path.join(PROJECT_DATASET_ROOT, self.dataset_name, self.model_name)
        print(f"Initializing HotpotQA Paraphrase Dataset: reasoning={self.reasoning}, debug={self.debug}, dataset_name={self.dataset_name}")
        super().__init__(self.dataset_name, model_name, paraphrase_file)
        
        if reasoning:
            self._instruction = \
                "Think through the given question, then output only the final answer.\n" + \
                "Output format (strict): ### Answer: <final answer>\n" + \
                "Do not include any explanation, reasoning, citations, or extra text—only the single answer line."
        else:
            self._instruction = "Answer the question based on the provided context in one or two sentences."
        
    @property
    def dataset_path(self):
        return os.path.join(self.dataset_root, "paraphrases_dataset")

    @property
    def instruction(self):
        return self._instruction

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
            assert isinstance(item['auto_paraphrases'], list), f"Expected list for auto_paraphrases, got {type(item['auto_paraphrases'])}"
            auto_paraphrases = item['auto_paraphrases']
            items.append({
                "uuid": uuid,
                "answers": answers,
                "manual_paraphrases": manual_paraphrases,
                "auto_paraphrases": auto_paraphrases
            })

            if self.debug and idx >= 199:
                break

        print(f"✓ Processed {len(items)} items")
        
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
        is_origs = []
        for item in batch:
            uuid = item["uuid"]
            random.seed(uuid)

            if self.additional_paraphrases is not None:
                assert uuid in self.additional_paraphrases, f"⚠️ Hotpot uuid {uuid} not found in additional paraphrases file"
                _paraphrases = [self.additional_paraphrases[uuid]["seed_prompt"]] + self.additional_paraphrases[uuid]["auto_paraphrases"]
                _is_origs = [True] + [False] * len(self.additional_paraphrases[uuid]["auto_paraphrases"])
                _paraphrases = _paraphrases[: self.num_paraphrases + 1]
                _is_origs = _is_origs[: self.num_paraphrases + 1]
            else:
                manual_list = item["manual_paraphrases"]
                auto_list = item["auto_paraphrases"][:self.num_paraphrases]
                _paraphrases = manual_list + auto_list
                _is_origs = [True] + [False] * self.num_paraphrases
                assert len(_paraphrases) == 1 + self.num_paraphrases, f"⚠️ Hotpot uuid {uuid}: total paraphrases {len(_paraphrases)} != {1 + self.num_paraphrases}"
            paraphrases.append(_paraphrases)
            is_origs.append(_is_origs)
        return uuids, answers, list(zip(*paraphrases)), list(zip(*is_origs))

    def get_few_shot_examples(self, k=5, seed=42):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}. Please run the dataset preparation first.")
        if k == 0:
            return "" 
        
        full_ds = load_from_disk(self.dataset_path)
        random.seed(seed)
        indices = random.sample(range(len(full_ds)), k)
        return "\n\n".join(self.format_example(full_ds[i]) for i in indices)

    def format_example(self, example):
        question = example["manual_paraphrases"][0] if isinstance(example["manual_paraphrases"], list) else example["manual_paraphrases"]
        answer = example["answers"][0] if isinstance(example["answers"], list) else example["answers"]
        return f"Q: {question}\nA: {answer}"
    
def get_dataset_instance(
        dataset_name, model_name, 
        debug=False, reasoning=False,
        num_paraphrases=None,
        additional_paraphrases_file=None):
    if dataset_name == "webqa":
        from dataset import WebQADataset
        dataset = WebQADataset(model_name=model_name)
    elif dataset_name == "myriadlama":
        from dataset import MyriadLamaDataset
        dataset = MyriadLamaDataset(
            model_name=model_name, 
            debug=debug, 
            num_paraphrases=num_paraphrases,
            paraphrase_file=additional_paraphrases_file)
    elif dataset_name == "commonsense":
        from dataset import CommonsenseParaphraseDataset
        dataset = CommonsenseParaphraseDataset(
            model_name=model_name, 
            debug=debug, 
            num_paraphrases=num_paraphrases,
            paraphrase_file=additional_paraphrases_file)
    elif dataset_name == "mmlu":
        from dataset import MMLUParaphraseDataset
        dataset = MMLUParaphraseDataset(
            model_name=model_name, 
            debug=debug, 
            num_paraphrases=num_paraphrases,
            paraphrase_file=additional_paraphrases_file)
    elif dataset_name == "logiqa":
        from dataset import LogiQAParaphraseDataset
        dataset = LogiQAParaphraseDataset(
            model_name=model_name, 
            debug=debug, 
            num_paraphrases=num_paraphrases,
            paraphrase_file=additional_paraphrases_file)
    elif dataset_name == "hotpot":
        from dataset import HotpotDataset
        dataset = HotpotDataset(
            model_name=model_name, 
            debug=debug, 
            reasoning=reasoning,
            num_paraphrases=num_paraphrases,
            paraphrase_file=additional_paraphrases_file)
    else:
        raise ValueError("Unsupported dataset. Please use 'webqa', 'myriadlama', 'commonsense', 'mmlu', 'logiqa', or 'hotpot'.")
    
    if model_name.startswith("phi3") and dataset.is_multi_choice:
        dataset.choice_labels = [label.strip() for label in dataset.choice_labels]
    
    return dataset

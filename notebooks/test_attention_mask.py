from bdb import set_trace
import pandas as pd
from tqdm import tqdm
from dataset import MyriadLamaDataset

dataset = MyriadLamaDataset(model_name="llama3.2_3b_it")
dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/net/tokyo100-10g/data/str01_01/xzhao/models/llama_hf/llama3.2_3b_it"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="cuda:0", torch_dtype="auto"
)
few_shot_examples = dataset.get_few_shot_examples(k=1)

sample_count = 0
for uuids, answers, all_paraphrases in tqdm(dataloader):
    batch_predictions = []
    batch_generations = []
    batch_templates = []

    # Process each question in batch
    for i, paraphrases in enumerate(zip(*all_paraphrases)):
        # All paraphrases in MyriadLAMA are manually generated
        # Simply select the first N paraphrases
        all_templates = list(paraphrases)
        selected_templates = all_templates[: 2]

        prompt, segment_metadata = dataset.construct_prompts_single_para_qapair(
            few_shot_examples, paraphrases=selected_templates
        )
        # prompt, segment_metadata = dataset.construct_prompts_with_paraphrases(
        #     few_shot_examples, paraphrases=selected_templates
        # )
        break
    break


# generation = myriadlama_flex_generation(prompt, segment_metadata, max_new_tokens=10)
import torch
from transformers import BatchEncoding
from generate_myriadlama2 import FlexAttentionWrapper
from generate_myriadlama2 import tokenize_with_segment
from generate_myriadlama2 import create_myriadlama_mask_mod

tokenizer.pad_token_id = tokenizer.eos_token_id
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.pad_token_id = tokenizer.eos_token_id

paraphrases = []
for i, length in enumerate(segment_metadata["len_paras"]):
    start = segment_metadata["len_context"] + sum(segment_metadata["len_paras"][:i])
    end = start + length
    paraphrases.append(prompt[start:end])

# Process prompt with position tracking and metadata
concatenated_text, full_tokens, segment_positions, original_length = (
    tokenize_with_segment(prompt, tokenizer, segment_metadata)
)

inputs = {
    "input_ids": torch.tensor([full_tokens]), 
    "attention_mask": torch.ones(1, len(full_tokens))
}
inputs = BatchEncoding(data=inputs).to(model.device)

modify_rope = False
if modify_rope:
    position_ids = torch.arange(len(full_tokens), dtype=torch.long, device=model.device)
    context_end = segment_positions[0]['end']
    start_generation_token_id = context_end + max(segment['end'] - segment['start'] for segment in segment_positions[1:])
    for segment in segment_positions[1:]:
        position_ids[segment['start']:segment['end']] = torch.arange(
            0, segment['end'] - segment['start'], dtype=torch.long, device=model.device
        ) + position_ids[context_end - 1] + 1

    position_ids = position_ids.unsqueeze(0).expand_as(inputs["input_ids"]) 
else:
    position_ids = torch.ones(len(full_tokens), dtype=torch.long, device=model.device)
    position_ids = position_ids.unsqueeze(0).expand_as(inputs["input_ids"]) 
    start_generation_token_id = original_length

inputs = BatchEncoding(data=inputs).to(model.device)
flex_wrapper = FlexAttentionWrapper(model)

print(concatenated_text)

generated = None
mask_mod = create_myriadlama_mask_mod(
    segment_positions, segment_metadata, original_length
)

# Generation loop
for step in range(10):
    flex_wrapper.patch_model(mask_mod)
    try:
        logits = model(inputs["input_ids"], attention_mask=inputs["attention_mask"], position_ids=position_ids).logits[:, -1, :]
    except Exception as e:
        import traceback
        print(f"⚠️  Generation step {step} failed: {type(e).__name__}: {e}")
        print(f"    Traceback:")
        traceback.print_exc()
        print(f"    Falling back to unpatched model...")
        flex_wrapper.unpatch_model()
        logits = model(inputs["input_ids"]).logits[:, -1, :]
    finally:
        # Always unpatch after each step
        flex_wrapper.unpatch_model()

    next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)

    if generated is None:
        generated = next_token
    else:
        generated = torch.cat([generated, next_token], dim=1)

    # Debug: show what was generated
    decoded_token = tokenizer.decode(next_token[0], skip_special_tokens=False)
    print(
        f"  Step {step}: generated token '{decoded_token}' (id: {next_token.item()})"
    )

    # Check for EOS or newline (likely end of one-word answer)
    if next_token.item() == tokenizer.eos_token_id:
        print(f"  Stopped: EOS token")
        break
    # Also check if we generated a newline or space (end of word)
    if "\n" in decoded_token and step > 0:  # Allow at least one token
        print(f"  Stopped: newline detected")
        break


# import pandas as pd
# from tqdm import tqdm
# from dataset import MyriadLamaDataset
# from generate_myriadlama import get_few_shot_examples_with_paraphrases

# dataset = MyriadLamaDataset(model_name="llama3.1_3b_it")
# dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
# few_shot_examples = get_few_shot_examples_with_paraphrases(dataset)


# from generate_myriadlama import construct_prompt_new_format


# for uuids, answers, all_paraphrases in tqdm(dataloader):
#     batch_predictions = []
#     batch_generations = []
#     batch_templates = []

#     # Process each question in batch
#     for i, paraphrases in enumerate(zip(*all_paraphrases)):
#         # All paraphrases in MyriadLAMA are manually generated
#         # Simply select the first N paraphrases
#         all_templates = list(paraphrases)
#         selected_templates = all_templates[: 5]

#         # Construct ONE prompt with ALL question paraphrases (NEW FORMAT)
#         # Prompt has: instruction + few-shot examples + ALL main question paraphrases
#         prompt = construct_prompt_new_format(
#             dataset.instruction,
#             few_shot_examples,
#             selected_templates,  # Pass ALL paraphrases, not just one
#         )
#         break
#     break

# from generate_myriadlama import concatenate_paraphrases_with_positions
# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_path = "/net/tokyo100-10g/data/str01_01/xzhao/models/llama_hf/llama3.2_3b_it"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# concatenated_text, segment_positions, segment_metadata, original_length = (
#     concatenate_paraphrases_with_positions(prompt, tokenizer)
# )

# from generate_myriadlama import create_myriadlama_mask

# mask_mod = create_myriadlama_mask(
#         segment_positions, segment_metadata, original_length
# )

# from generate_myriadlama import FlexAttentionWrapper

# model = AutoModelForCausalLM.from_pretrained(
#     model_path, device_map="cuda:0", torch_dtype="auto"
# )
# flex_wrapper = FlexAttentionWrapper(model)
# flex_wrapper.patch_model(mask_mod)
# inputs = tokenizer(
#     concatenated_text, return_tensors="pt", truncation=True, add_special_tokens=True
# ).to(model.device)

# logits = model(inputs["input_ids"]).logits[:, -1, :]

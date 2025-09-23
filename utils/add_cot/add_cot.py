import json
import requests
import time
from tqdm import tqdm
import threading
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from deepseek import  call_deepseek_R1
def extract_messages(messages):
    system_prompt = ""
    user_input = ""
    assistant_output = ""
    
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = "You are a scientific writing assistant. Produce a structured outline based on the article metadata and references provided.You need to pay attention to the differences in review structure between different disciplines. For example, medical reviews should follow the IMRaD structure, while computer science reviews should be written in a taxonomic structure or in the order of theoretical development."
        elif msg["role"] == "user":
            user_input = msg["content"]
        elif msg["role"] == "assistant":
            assistant_output = msg["content"]
    
    return system_prompt, user_input, assistant_output

def generate_thought_chain(title, reference, outline):
        prompt = f"""
        Based on the given review outline and references, describe the reasoning process for deriving the outline from the references.
        This reasoning process should consist of only two steps.
        1. Cluster the references into categories based on corresponding themes. These categories should be similar to the subheadings that appear in the actual outline. When describing the references, use  text rather than numbers.
        2. Describe the process for generating the complete outline from these themes.
        Title:{title}
        References:{reference}
        Target Outline:{outline}
        The "ref" field in the "outline" field is the actual reference. The references you mention should be as consistent as possible with it, rather than using other references in the "reference" field.
        """
        system_content = "You are an expert in review writing. You need to show your complete reasoning when you write an outline."
        return call_deepseek_R1(system_content, prompt)

    
def worker(item):
    title = item["meta"]["title"]
    reference = item["ref_meta"]
    outline = item["outline"]
    try:
        answer, thought_chain = generate_thought_chain(title, reference, outline)
    except Exception as e:
        return None, None, e
    return answer, thought_chain, None

def process_data_parallel(input_file, output_file, max_workers=32):
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            data.append(obj)
    results = [None] * len(data)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, item): idx for idx, item in enumerate(data)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
            idx = futures[future]
            answer, thought_chain, error = future.result()
            if error:
                print(f"Item {idx} processing error: {error}")
            elif thought_chain:
                results[idx] = (answer, thought_chain)

    for idx, res in enumerate(results):
        if res:
            answer, thought_chain = res
            item = data[idx]
            item["cot"] = answer
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            if "cot" in item:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f" save as {output_file}")

if __name__ == "__main__":
    process_data_parallel("data_abs.jsonl", "data_thinking.jsonl")
import json
import random
from collections import Counter
from datetime import datetime

def del_ref(data):
    for item in data:
        item['ref'] = []
        data[data.index(item)] = item
    return data

def classify_and_count(data):
    categories_counter = Counter()
    
    for item in data:
        categories = item.get('meta', {}).get('categories', '')
        
        if categories:
            category = categories.split(' ')[0]
            category = category.split('.')[0]
            categories_counter[category] += 1
    
    print("="*10)
    for category, count in categories_counter.items():
        print(f"{category}: {count}")


USER_PROMPT_TEMPLATE = """
Write an outline for a literature review based on the given title and references.
If I set Output_references to False, do not generate the contents of the ref list for me
Format: {{"level": 1, "numbering": "1", "title": "Introduction", "ref": ["key1","key2"...]}}
Output_references: True
Title:
{title}
References:
{references_json}
"""

SYSTEM_PROMPT = (
    "You are a scientific writing assistant. Produce a structured outline based on the article metadata and references provided."
)

def load_jsonl(file_path):
    """读取jsonl文件并返回数据列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

def filter_data(data):
    """筛选出meta.update_date在2025年之后的数据，并返回剩余数据"""
    filtered_data = []
    remaining_data = []
    
    for item in data:
        update_date = item.get('meta', {}).get('update_date', '')
        #print(update_date)
        try:
            update_date_obj = update_date[0:4]
            if int(update_date_obj) > 2024:
                filtered_data.append(item)
            else:
                remaining_data.append(item)
        except ValueError:
            continue
    if len(filtered_data) > 100:
        filtered_data = filtered_data[:100]
        print("2025年数据大于100")
    return filtered_data, remaining_data

def split_data(remaining_data):
    random.shuffle(remaining_data)
    split_index = int(len(remaining_data) * 0.7)
    return remaining_data[:split_index], remaining_data[split_index:]

def convert_data_to_format(data, add_cot=False):
    formatted_data = []
    for item in data:
        message = {
            "messages": [
                {
                    "content": SYSTEM_PROMPT,
                    "role": "system"
                },
                {
                    "content": USER_PROMPT_TEMPLATE.format(title=item['meta']['title'] , references_json=item['ref_meta']),
                    "role": "user"
                },
                {
                    "content": str(item['outline']),
                    "role": "assistant"
                }
            ]
        }
        if add_cot:
            cot_content = f"<think>{item['cot']}</think>"
            message['messages'][2]['content'] = cot_content + message['messages'][2]['content']
        
        formatted_data.append(message)
    return formatted_data

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
def save_ids_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            meta_id = item.get("meta", {}).get("id", None)
            if meta_id is not None:
                f.write(json.dumps({"id": meta_id}, ensure_ascii=False) + "\n")
def main(input_file):
    # 读取数据
    data = load_jsonl(input_file)

    # 筛选出2025年之后的数据以及剩余数据
    filtered_data, remaining_data = filter_data(data)

    filtered_data_formatted = convert_data_to_format(filtered_data)

    save_json(filtered_data_formatted, 'test100.json')


if __name__ == '__main__':
    input_file = 'test.jsonl'  # input jsonl file
    main(input_file)

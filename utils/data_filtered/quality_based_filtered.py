import json
# quality-based filtering
# This code is used for filtering data based on outline quality
# This code requires the input JSON file to have an outline field
# Each outline entry should be a dictionary with "level" and "title" keys

def truncate_outline(outline):
    """
    Cut off the outline at certain keywords
    """
    # Define the set of keywords that need to be truncated
    truncate_keywords = ["conclusions", "summary", "concluding", "conclusion", 'general discussion']
    remove_keywords = ['acknowledgement', 'appendix', "acknowledgements","acknowledgment"]
    truncate_keywords = [keyword.lower() for keyword in truncate_keywords]
    remove_keywords = [keyword.lower() for keyword in remove_keywords]
    for i, item in enumerate(outline):
        if item.get("level") == 1:
            title_lower = item.get("title", "").lower()
            if any(keyword in title_lower for keyword in remove_keywords):
                return outline[:i]
            if any(keyword in title_lower for keyword in truncate_keywords):
                end_index = len(outline)
                for j in range(i+1, len(outline)):
                    if outline[j].get("level") == 1:
                        end_index = j
                        break
                return outline[:end_index]
    return outline
def is_valid_outline(data):
    outline = data.get("outline", [])
    if len(outline) < 6:
        return False


    if any(item.get("level") == 4 for item in outline):
        print("The outline includes four level headings:", outline)
        return False
    level1_count = sum(1 for item in outline if item["level"] == 1)
    if level1_count < 4:
        print("Too few sections")
        return False
    first_title = outline[0]["title"].lower() if outline else ''
    allowed_keywords = ['introduction', 'background']
    if not any(keyword in first_title for keyword in allowed_keywords):
        print(f"First Title '{outline[0]['title']}' not include {allowed_keywords}")
        return False
    
    # Check if the title length is too short
    for item in outline:
        if len(item["title"]) < 3:
            print(f"Title '{item['title']}' too short")
            return False
    # Skip outlines with only top-section
    if all(item["level"] == 1 for item in outline):
        return False
    level1_count = sum(1 for item in outline if item["level"] == 1)
    if level1_count < 4:
        return False    
    keywords = ["acknowledgments", "appendix"]
    if any(any(keyword.lower() in item["title"].lower() for keyword in keywords) for item in outline):
        return False
        
        
    #  Check the number of subheadings within the section
    level1_count = 0
    for item in outline:
        if item["level"] == 1:
            level1_count += 1
            if level1_count > 10:  
                return False
        else:
            level1_count = 0
    
    #  Check for duplicate section names
    titles = [item["title"].lower() for item in outline if item["level"] == 1]
    if len(titles) != len(set(titles)):
        return False
        
    
    return True


def process_outline_to_str(data):
    if not data.get("outline"):
        return ""
    processed_items = [
        f"{item['numbering']} {item['title']}" 
        for item in data["outline"]
    ]
    
    return "\n".join(processed_items)

def process_file(input_path, output_path):
    skipped_count = 0
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            try:
                data = json.loads(line.strip())

                if "outline" in data and isinstance(data["outline"], list):
                    data["outline"] = truncate_outline(data["outline"])
                else:
                    print("no outline:", data)
                
                if len(data["ref_meta"])<=10 or len(data["ref_meta"])>600:
                    print("references too few:", len(data["ref_meta"]))
                    skipped_count += 1
                    continue
                if not is_valid_outline(data):
                    skipped_count += 1
                    continue
                processed_outline = process_outline_to_str(data)
            
                data["outline_str"] = processed_outline
                
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            except json.JSONDecodeError as e:
                print(f"Error: {e} ")
    print(f"totle skip {skipped_count} low quality outlines")


input_files = "data_medbiov3.jsonl"
output_files = "data_medbiov4.jsonl"

process_file(input_files, output_files)
print(f"save at: {output_files}")
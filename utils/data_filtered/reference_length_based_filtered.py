import json
import sys

def filter_references(data):
    """
    Filter out ref_meta entries where the title has fewer than 10 English letters
    or contains specific keywords
    """
    removed_titles = []
    if "ref_meta" not in data or not isinstance(data["ref_meta"], list):
        return data
    
    kept = []
    for ref in data["ref_meta"]:
        if not isinstance(ref, dict) or "title" not in ref:
            continue

        title = ref["title"]
        if not isinstance(title, str):
            continue

        # Count English letters in title
        letter_count = sum(1 for c in title if c.isalpha())
        if letter_count >= 10 or any(keyword in title for keyword in ['GPT', 'B', 'Q-learning', 'Openai', 'nn']):
            kept.append(ref)
        else:
            removed_titles.append(title)

    data["ref_meta"] = kept 
    return data, removed_titles

def process_jsonl_file(input_path, output_path, output_bad_path="bad_articles.jsonl"):
    """
    Process JSONL file by:
    - Skipping records with reflen < 10
    - Filtering ref_meta entries with title letter count < 10
    - Calculating and printing filtering ratio per article
    - Generating and printing global statistics
    """
    skipped_count = 0
    total_count = 0
    filtered_count = 0
    bad_article_count = 0
    bad_records = []

    try:
        with open(input_path, "r", encoding="utf-8") as infile, \
             open(output_path, "w", encoding="utf-8") as outfile:

            for idx, line in enumerate(infile, 1):
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    print(f"[Line {idx}] Invalid JSON, skipping", file=sys.stderr)
                    continue

                # Check reflen
                if len(data["ref_meta"]) < 5:
                    skipped_count += 1
                    continue

                # Original reference count
                original_ref_count = len(data.get("ref_meta", []))

                # Filter ref_meta entries
                data, removed_titles = filter_references(data)
                new_ref_count = len(data.get("ref_meta", []))

                # Calculate filtering ratio for this article
                filtered_for_this = original_ref_count - new_ref_count
                ratio_for_this = (filtered_for_this / original_ref_count) if original_ref_count > 0 else 0.0

                # Print filtering ratio for this article
                article_id = data.get("id", f"line{idx}")
                if ratio_for_this > 0.05:
                    print(f"Article {article_id}: original {original_ref_count} refs, filtered {filtered_for_this} refs, "
                          f"ratio {ratio_for_this:.2%}")

                # Update global statistics
                filtered_count += filtered_for_this
                data["reflen"] = new_ref_count

                # Write to main output
                if new_ref_count >= 5:
                    total_count += 1
                    outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

                    # Mark as problematic article if any references were filtered
                    if filtered_for_this > 0:
                        bad_article_count += 1
                        bad_records.append({
                            "id": article_id,
                            "title": data["meta"]["title"],
                            "removed_ref_titles": removed_titles
                        })

        # Write problematic articles to separate file
        if output_bad_path and bad_records:
            with open(output_bad_path, "w", encoding="utf-8") as badfile:
                for rec in bad_records:
                    badfile.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Print global statistics
        print(f"\nOverall Statistics:")
        print(f"  Valid articles written: {total_count}")
        print(f"  Skipped articles (reflen < 5): {skipped_count}")
        print(f"  Total references filtered: {filtered_count}")
        print(f"  Problematic articles: {bad_article_count}")
        if total_count > 0:
            overall_ratio = bad_article_count / total_count
            print(f"  Problematic article ratio: {overall_ratio:.2%} ({bad_article_count}/{total_count})")
        else:
            print("  No valid records processed.")

    except FileNotFoundError:
        print(f"Input file not found: {input_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)

if __name__ == "__main__":
    input_file = "input.jsonl"
    output_file = "output.jsonl"
    output_bad_file = "bad_articles.jsonl"
    process_jsonl_file(input_file, output_file, output_bad_file)
import json


# This code checks if references in the outline all appear in ref_meta
def filter_and_save_jsonl(input_path, output_path, threshold=0.01):
    """
    Read the JSONL file specified by input_path line by line. For each record:
      1. Collect outline_refs and ref_meta_keys
      2. Calculate missing references: missing_refs = outline_refs - ref_meta_keys
      3. If missing_refs is not empty and its ratio < threshold:
           - Remove these missing_refs from the outline
      4. Write the (possibly modified) record to the output_path file
    Also prints processing statistics.
    """
    total_records = 0
    modified_records = 0
    dropped_records = 0
    untouched_records = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            total_records += 1

            record = json.loads(line)

            # Collect outline_refs
            outline_refs = {
                ref_key
                for section in record.get("outline", [])
                if isinstance(section.get("ref", []), list)
                for ref_key in section["ref"]
            }

            # Collect ref_meta_keys
            ref_meta_keys = {
                entry.get("key")
                for entry in record.get("ref_meta", [])
                if entry.get("key") is not None
            }

            missing_refs = outline_refs - ref_meta_keys

            # Handle missing references with ratio check
            if missing_refs and len(outline_refs) > 0:
                ratio = len(missing_refs) / len(outline_refs)
                if ratio < threshold:
                    # Minor missing references: remove these keys and keep the record
                    for section in record["outline"]:
                        if isinstance(section.get("ref", []), list):
                            section["ref"] = [
                                rk for rk in section["ref"]
                                if rk not in missing_refs
                            ]
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    modified_records += 1
                else:
                    # Discard the record due to significant missing references
                    dropped_records += 1
            else:
                # No missing references or outline is empty: write as-is
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                untouched_records += 1

    # Print statistics
    print(f"Total records: {total_records}")
    print(f"Records with minor missing references (<{threshold*100:.1f}%) and trimmed: {modified_records}")
    print(f"Records discarded due to significant missing references (≥{threshold*100:.1f}%): {dropped_records}")
    print(f"Records untouched (no missing references): {untouched_records}")

if __name__ == "__main__":
    # Example usage
    filter_and_save_jsonl(
        input_path="input.jsonl",
        output_path="output.jsonl",
        threshold=0.01
    )
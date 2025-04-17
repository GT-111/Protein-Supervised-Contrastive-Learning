import os
import requests
import random
import time
from tqdm import tqdm

def query_pdb_ids_by_struct_title(keyword, count):
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "struct.title",
                "operator": "contains_words",
                "value": keyword
            }
        },
        "return_type": "entry",
        "request_options": {
            "return_all_hits": True
        }
    }

    response = requests.post(url, json=query)
    if response.status_code != 200:
        print("Error:", response.json())
        return []

    results = response.json()
    return [item['identifier'] for item in results.get('result_set', [])][:count]

def download_pdb_file(pdb_id, save_dir, subclass, max_retries=3):
    os.makedirs(save_dir, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(os.path.join(save_dir, f"{pdb_id}.pdb"), 'w') as f:
                    f.write(response.text)
                with open("id_subclass_map.txt", "a") as f:
                    f.write(f"{pdb_id.upper()}\t{subclass}\n")
                return True
            else:
                print(f"[Attempt {attempt}] Failed to download {pdb_id}: Status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[Attempt {attempt}] Network error for {pdb_id}: {e}")
        time.sleep(2)
    print(f"❌ Failed to download {pdb_id} after {max_retries} attempts.")
    return False

def download_fixed_number(pdb_pool, target_count, save_dir, subclass_lookup):
    downloaded = set()
    attempted = set()
    while len(downloaded) < target_count and len(attempted) < len(pdb_pool):
        candidates = list(set(pdb_pool) - attempted)
        random.shuffle(candidates)
        for pdb_id in tqdm(candidates, desc=f"Downloading to {save_dir} ({len(downloaded)}/{target_count})"):
            attempted.add(pdb_id)
            subclass = subclass_lookup(pdb_id)
            if download_pdb_file(pdb_id, save_dir, subclass):
                downloaded.add(pdb_id)
            if len(downloaded) >= target_count:
                break
            
if __name__ == "__main__":
    if os.path.exists("id_subclass_map.txt"):
        os.remove("id_subclass_map.txt")

    # Step 1: Enzymes
    enzyme_keywords = ["hydrolase", "oxidoreductase", "isomerase", "transferase", "lyase", "ligase"]
    all_enzyme_ids = []
    keyword_source_enzyme = {}
    for kw in enzyme_keywords:
        ids = query_pdb_ids_by_struct_title(kw, 300)
        for pid in ids:
            keyword_source_enzyme[pid.upper()] = kw
        all_enzyme_ids += ids
    all_enzyme_ids = list(set(all_enzyme_ids))
    enzyme_sampled = random.sample(all_enzyme_ids, min(1000, len(all_enzyme_ids)))
    print(f"Randomly selected {len(enzyme_sampled)} enzyme PDB IDs.")
    download_fixed_number(enzyme_sampled, 1000, "data/enzymes", subclass_lookup=lambda x: keyword_source_enzyme.get(x.upper(), "unknown"))

    # Step 2: Non-enzymes
    non_enzyme_keywords = ["structural protein", "receptor", "channel", "signaling", "membrane protein", "DNA-binding"]
    all_non_ids = []
    keyword_source_non = {}
    for kw in non_enzyme_keywords:
        ids = query_pdb_ids_by_struct_title(kw, 400)
        for pid in ids:
            keyword_source_non[pid.upper()] = kw
        all_non_ids += ids
    all_non_ids = list(set(all_non_ids))
    non_enzyme_sampled = random.sample(all_non_ids, min(2000, len(all_non_ids)))
    print(f"Randomly selected {len(non_enzyme_sampled)} non-enzyme PDB IDs.")
    download_fixed_number(non_enzyme_sampled, 2000, "data/non_enzymes", subclass_lookup=lambda x: keyword_source_non.get(x.upper(), "unknown"))

    print("✅ All downloads complete and mapping saved to id_subclass_map.txt.")

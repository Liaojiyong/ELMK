import argparse
import json
import jsonlines
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def split_to_samples_and_map(obj):
    if isinstance(obj, dict):
        return list(obj.keys()), obj
    if isinstance(obj, (list, set, tuple)):
        return list(obj), {}
    try:
        return list(obj), {}
    except Exception:
        raise TypeError(f"Unsupported split type: {type(obj)}. Expect dict/list/set/tuple.")

def normalize_fields(meta):
    types = meta.get("types")
    entity_set = meta["Entity_set"]   
    label = meta["Label"]            
    return types, entity_set, label

def build_global_index(*maps):
    merged = {}
    for m in maps:
        if m:
            merged.update(m)
    return merged

def write_split(split_name, samples, preferred_map, fallback_index, out_path):
    out_path = Path(out_path)
    with jsonlines.open(out_path, mode="w") as writer:
        for i, q in enumerate(samples, start=1):
            meta = preferred_map.get(q) or fallback_index.get(q)
            if meta is None:
                types, entity_set, label = None, None, None
            else:
                types, entity_set, label = normalize_fields(meta)

            writer.write({
                "question_id": i,
                "question": q,
                "types": types,
                "entity_set": entity_set,
                "Label": label,
            })

def main():
    parser = argparse.ArgumentParser(description="Extract FactKG splits to JSONL.")
    parser.add_argument("--factkg_train", type=str, default='./factkg_train.pickle', help="Path for factkg train set.")
    parser.add_argument("--factkg_dev", type=str,  default='./factkg_dev.pickle', help="Path for factkg dev set.")
    parser.add_argument("--factkg_test", type=str, default='./factkg_test.pickle', help="Path for factkg test set.")
    parser.add_argument("--on_missing", type=str, choices=["empty", "skip"], default="empty", help="If metadata for a sample is missing: 'empty'=write None placeholders; 'skip'=drop it.")
    args = parser.parse_args()

    train_obj = load_pickle(args.factkg_train)
    dev_obj = load_pickle(args.factkg_dev)
    test_obj = load_pickle(args.factkg_test)
    train_samples, train_map = split_to_samples_and_map(train_obj)
    dev_samples, dev_map = split_to_samples_and_map(dev_obj)
    test_samples, test_map = split_to_samples_and_map(test_obj)
    global_index = build_global_index(train_map, dev_map, test_map)
    write_split("train", train_samples, train_map, global_index, out_path="factkg_train_set.jsonl")
    write_split("dev", dev_samples, dev_map, global_index, out_path="factkg_dev_set.jsonl")
    write_split("test", test_samples, test_map, global_index, out_path="factkg_test_set.jsonl")

if __name__ == "__main__":
    main()

import os, sys
import ast
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import time
import pickle
import argparse
from typing import List, Any, Dict, Iterable, Set, Tuple
import numpy as np
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from transformers import AutoTokenizer
from openai import OpenAI
from encoder.pretrain_encoder import KGQAEncoder
from tqdm import tqdm
from metric import MetricsTracker

os.environ["TOKENIZERS_PARALLELISM"] = "false"
encode_batch_size = int(os.environ.get("encode_batch_size", 64))
max_seq_len = int(os.environ.get("max_seq_len", 128))
torch.backends.cuda.matmul.allow_tf32 = True
openai_key_file = "../openai_api_key.txt"
kg_file = "../data/metaqa_kg.pickle"
llm_model = "gpt-4o-mini"
max_tokens = 400
temperature = 0.2
top_p = 0.1

parser = argparse.ArgumentParser(description="Parsing input arguments.")
parser.add_argument("--question_model", type=str, default=None, required=False, help="Path to question encoder checkpoint (auto-pick by hop if omitted)")
parser.add_argument("--hop", type=int, required=True, choices=[1, 2, 3], help="Number of hops")
parser.add_argument("--k", type=int, required=True, help="The Top-K path to be selected depends on the hop, please refer to allowed_k_by_hop dictionary")
parser.add_argument("--model_path", type=str, default="../encoder/sentence-transformers/all-mpnet-base-v2")
args = parser.parse_args()
hop = args.hop
k = args.k
model_path = args.model_path

allowed_k_by_hop = {
    1: [5, 10, 15, 20, 25],
    2: [10, 20, 30, 40, 50],
    3: [60, 70, 80, 90, 100],
}
if k not in allowed_k_by_hop[hop]:
    raise SystemExit(f"Invalid --k={k} for hop={hop}. Allowed: {allowed_k_by_hop[hop]}")

prompt_file = f"../prompts/test_prompts/{hop}_hop_prompts.txt"
log_file = Path(f"{hop}_hop_metrics_log.txt")
log_file.parent.mkdir(parents=True, exist_ok=True)
log_file.touch(exist_ok=True)

if args.question_model:
    question_model_path = args.question_model
else:
    question_model_path = f"../encoder/checkpoint/{hop}-hop-model.pth"
if not os.path.exists(question_model_path):
    raise SystemExit(f"Question model not found: {question_model_path}")

if hop == 1:
    question_data = "../data/1_hop_test.jsonl"
elif hop == 2:
    question_data = "../data/2_hop_test.jsonl"
else:
    question_data = "../data/3_hop_sample3000.jsonl"

def open_file(filepath):
    with open(filepath, "r", encoding="utf-8") as infile:
        return infile.read()

with open(kg_file, "rb") as f:
    kg = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained(model_path)
question_model = KGQAEncoder(model_path)
question_model.load_state_dict(torch.load(question_model_path, weights_only=True))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
question_model.to(device)
question_model.eval()

client = OpenAI(
    api_key=open_file(openai_key_file),
    base_url="https://xh.v1api.cc/v1",
)

def build_subgraph(entity_set, knowledge_graph, max_hop):
    subgraph = set()
    for s in entity_set:
        if s not in knowledge_graph:
            continue
        for r1, tails1 in knowledge_graph[s].items():
            for o1 in tails1:
                subgraph.add((str(s), str(r1), str(o1)))               # 1-hop
                if max_hop >= 2 and o1 in knowledge_graph:
                    for r2, tails2 in knowledge_graph[o1].items():
                        for o2 in tails2:                              # 2-hop
                            subgraph.add((str(s), str(r1), str(o1), str(r2), str(o2)))
                            if max_hop >= 3 and o2 in knowledge_graph:
                                for r3, tails3 in knowledge_graph[o2].items():
                                    for o3 in tails3:                  # 3-hop
                                        subgraph.add((str(s), str(r1), str(o1), str(r2), str(o2), str(r3), str(o3)))
    return subgraph

def triplet_to_text(path):
    n = len(path)
    if n == 3:                                      # 1-hop
        s, r, o = path
        if r and r[0] == "~":
            return f"{o} {r[1:]} {s}"
        return f"{s} {r} {o}"

    elif n == 5:                                    # 2-hop
        a, r1, b, r2, c = path
        if r1 and r1[0] == "~":
            text = f"{b} {r1[1:]} {a}, "
        else:
            text = f"{a} {r1} {b}, "
        if r2 and r2[0] == "~":
            text += f"{c} {r2[1:]} {b}"
        else:
            text += f"{b} {r2} {c}"
        return text

    elif n == 7:                                    # 3-hop
        a, r1, b, r2, c, r3, d = path
        if r1 and r1[0] == "~":
            text = f"{b} {r1[1:]} {a}, "
        else:
            text = f"{a} {r1} {b}, "
        if r2 and r2[0] == "~":
            text += f"{c} {r2[1:]} {b}, "
        else:
            text += f"{b} {r2} {c}, "

        if r3 and r3[0] == "~":
            text += f"{d} {r3[1:]} {c}"
        else:
            text += f"{c} {r3} {d}"
        return text
    return None

def select_paths_kmeans(paths, positive_embedding, question_embedding, k_top, n_clusters):
    num_paths = len(paths)
    if num_paths == 0:
        return []
    if num_paths <= k_top:
        return paths

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(positive_embedding)
    eps = 1e-12
    q_norm = question_embedding / (np.linalg.norm(question_embedding) + eps)
    path_norms = positive_embedding / (np.linalg.norm(positive_embedding, axis=1, keepdims=True) + eps)
    sim_scores = np.dot(path_norms, q_norm)
    q_val = k_top // n_clusters
    r_val = k_top % n_clusters
    selected_paths = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        if cluster_indices.size == 0:
            continue
        quota = q_val + (1 if cluster_id < r_val else 0)
        sorted_indices = sorted(cluster_indices, key=lambda idx: sim_scores[idx], reverse=True)
        selected_indices = sorted_indices[: min(quota, len(sorted_indices))]
        selected_paths.extend([paths[i] for i in selected_indices])
    return selected_paths

def encode_texts_in_batches(model, tokenizer, texts, device, batch_size, max_length):
    if len(texts) == 0:
        return np.zeros((0, 768), dtype=np.float32) 
    use_cuda = device.type == "cuda"
    results: List[np.ndarray] = []
    curr_bs = max(1, batch_size)
    idx = 0
    while idx < len(texts):
        end = min(idx + curr_bs, len(texts))
        batch_texts = texts[idx:end]
        try:
            with torch.inference_mode():
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )

                if use_cuda:
                    encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}
                    try:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            batch_emb = model(
                                input_ids=encoded["input_ids"],
                                attention_mask=encoded["attention_mask"],
                            )
                    except RuntimeError as e:
                        batch_emb = model(
                            input_ids=encoded["input_ids"],
                            attention_mask=encoded["attention_mask"],
                        )
                else:
                    batch_emb = model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                    )
                if isinstance(batch_emb, torch.Tensor):
                    results.append(batch_emb.detach().cpu().numpy())
                else:
                    if hasattr(batch_emb, "last_hidden_state"):
                        pooled = batch_emb.last_hidden_state[:, 0, :] 
                        results.append(pooled.detach().cpu().numpy())
                    else:
                        raise RuntimeError("Unexpected embedding output type.")
            idx = end
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and use_cuda:
                torch.cuda.empty_cache()
                curr_bs = max(1, curr_bs // 2)
                if curr_bs == 1:
                    device_cpu = torch.device("cpu")
                    model.to(device_cpu)
                    use_cuda = False
                continue
            else:
                raise
        finally:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
    embeddings = np.concatenate(results, axis=0)
    return embeddings
    
def run_llm_on_context(question_list, context_texts):
    prompt = open_file(prompt_file)
    prompt += f"Context : {context_texts[0]} Question : {question_list}\n"
    prompt += "Answer: "
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=60,
    )
    generated_text = response.choices[0].message.content
    usage = getattr(response, "usage", None)
    if usage is not None and isinstance(usage, dict):
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        tt = usage.get("total_tokens", 0)
    elif usage is not None:
        pt = getattr(usage, "prompt_tokens", 0)
        ct = getattr(usage, "completion_tokens", 0)
        tt = getattr(usage, "total_tokens", 0)
    else:
        pt = ct = tt = 0
    try:
        result_obj = ast.literal_eval(generated_text)
    except Exception:
        result_obj = set()
    return generated_text, result_obj, (pt, ct, tt)

def main():
    questions_dict = {}
    entity_set_dict = {}
    label_set_dict = {}
    with open(question_data, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            dataset = json.loads(line)
            qid = dataset["question_id"]
            questions_dict[qid] = dataset["question"]
            entity_set_dict[qid] = dataset["entity_set"]
            label_set_dict[qid] = dataset["Label"]

    dataset_len = len(questions_dict)
    data_num = range(1, dataset_len + 1)
    tracker = MetricsTracker(log_file=str(log_file), print_each_batch=True)
    t0 = time.time()
    question_id_list = []
    question_list = []
    entity_set_list = []
    ground_truth_list = []
    contexts_list = []

    for processed, num in enumerate(tqdm(data_num), start=1):
        question = questions_dict[num]
        entity_set = entity_set_dict[num]
        ground_truth = label_set_dict[num]
        subgraph = list(build_subgraph(entity_set, kg, max_hop=hop))   
        path_texts: List[str] = []                                    
        for path in subgraph:
            pos = triplet_to_text(path)
            if pos is not None:
                path_texts.append(pos)
        with torch.inference_mode():
            encoded_q = tokenizer(question, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt")
            if device.type == "cuda":
                encoded_q = {k: v.to(device, non_blocking=True) for k, v in encoded_q.items()}
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        question_embedding_tensor = question_model(input_ids=encoded_q["input_ids"], attention_mask=encoded_q["attention_mask"])
                except RuntimeError:
                    question_embedding_tensor = question_model(input_ids=encoded_q["input_ids"], attention_mask=encoded_q["attention_mask"])
            else:
                question_embedding_tensor = question_model(input_ids=encoded_q["input_ids"], attention_mask=encoded_q["attention_mask"])
        question_embedding = question_embedding_tensor.detach().cpu().numpy().flatten()

        if len(path_texts) > 0:                                                             
            positive_embedding = encode_texts_in_batches( question_model, tokenizer, path_texts, device, batch_size=encode_batch_size, max_length=max_seq_len)
        else:
            positive_embedding = np.zeros((0, question_embedding.size), dtype=np.float32)
        n_clusters = min(len(path_texts), 4) if len(path_texts) > 0 else 1                     
        selected_paths = select_paths_kmeans(path_texts, positive_embedding, question_embedding, k, n_clusters)
        context_texts = "[" + ", ".join(selected_paths) + "]"

        question_id_list.append(num)
        question_list.append(question)
        entity_set_list.append(entity_set)
        ground_truth_list.append(ground_truth)
        contexts_list.append(context_texts)
        if len(question_id_list) >= 1:
            _, result_obj, usage = run_llm_on_context(question_list, contexts_list)
            print('result_obj:', result_obj)
            tracker.on_batch(
                question_list=question_list,
                ground_truth_list=ground_truth_list,
                result_obj=result_obj,
                usage_tuple=usage,
                context_texts=contexts_list,
            )

            # reset buffers
            question_id_list.clear()
            question_list.clear()
            entity_set_list.clear()
            ground_truth_list.clear()
            contexts_list.clear()

        if processed % 500 == 0:
            elapsed_now = time.time() - t0
            tracker.periodic_log(step_tag=f"Step {processed // 500}", elapsed_seconds=elapsed_now)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_total = time.time() - t0
    tracker.final_log(elapsed_seconds=elapsed_total)
    print("Done.")


if __name__ == "__main__":
    main()
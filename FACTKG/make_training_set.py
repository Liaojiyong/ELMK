import os
import re
import json
import time
import random
import pickle
import argparse
from typing import Dict, List, Tuple, Iterable, Set, Any, Optional
from tqdm import tqdm
from openai import OpenAI

max_rel_candidates = 40       
max_objs_per_rel   = 20       
topk_rel_by_llm    = 5        
topk_ents_by_llm   = 5       

model_name   = "gpt-4o-mini"
max_tokens   = 400
temperature  = 0.2
top_p        = 0.1
retry_times  = 3
backoff_secs = 5
random.seed(42)

def open_file(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return f.read().strip()

def has_comma(x):
    return "," in str(x)

def llm(client, prompt, max_tokens):
    for attempt in range(retry_times):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            content = resp.choices[0].message.content
            return content
        except Exception as e:
            if attempt == retry_times - 1:
                print("[ERROR LLM]", e)
                return None
            time.sleep(backoff_secs)


pair_re = re.compile(r"\(\s*([^(),]+)\s*,\s*([^(),]+)\s*\)")
triple_re = re.compile(r"\(\s*([^(),]+)\s*,\s*([^(),]+)\s*,\s*([^(),]+)\s*\)")
def parse_pairs(text, k):
    if not text:
        return []
    pairs = [(a.strip().strip("'\" "), b.strip().strip("'\" "))
             for a, b in pair_re.findall(text)]
    dedup = []
    seen = set()
    for p in pairs:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
        if len(dedup) >= k:
            break
    return dedup

def parse_triples(text, k):
    if not text:
        return []
    out = []
    seen = set()
    for a, b, c in triple_re.findall(text):
        a = a.strip().strip("'\" ")
        b = b.strip().strip("'\" ")
        c = c.strip().strip("'\" ")
        tup = (a, b, c)
        if tup not in seen:
            seen.add(tup)
            out.append(tup)
            if len(out) >= k:
                break
    return out

ans_line_re = re.compile(r"answer\s*[:\-]?\s*(true|false)", re.IGNORECASE)
def parse_true_false_lines(text, n):
    if not text:
        return ["unknown"] * n
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    out = []
    for line in lines:
        m = ans_line_re.search(line)
        if m:
            out.append(m.group(1).lower())
    if len(out) < n:
        tf_all = re.findall(r"\b(true|false)\b", text, flags=re.IGNORECASE)
        for v in tf_all:
            if len(out) >= n:
                break
            out.append(v.lower())
    if len(out) < n:
        out += ["unknown"] * (n - len(out))
    else:
        out = out[:n]
    return out

def build_subgraph_rels_ranking(client, question, entity_set, kg):
    one_hop = get_one_hop_neighbors_rels_ranking(client, question, entity_set, kg)
    if one_hop == "No" or not one_hop:
        return "No"
    two_hop = get_two_hop_neighbors_rels_ranking(client, question, one_hop, kg)
    if two_hop == "No" or not two_hop:
        return "No"
    return one_hop + two_hop

def get_one_hop_neighbors_rels_ranking(client, question, entity_set, kg):
    neighbors_rel = set()                                         
    for ent in entity_set:
        if ent in kg:
            for rel in kg[ent].keys():
                if not has_comma(ent) and not has_comma(rel):
                    neighbors_rel.add((str(ent), str(rel)))
    if not neighbors_rel:
        return "No"

    rel_pool = list(neighbors_rel)                                 
    if len(rel_pool) > max_rel_candidates:
        rel_pool = random.sample(rel_pool, max_rel_candidates)

    top_pairs = rank_pairs_by_llm(client, question, rel_pool, topk_rel_by_llm, max_tokens)  
    neighbors_ent = []
    num_ent = 0
    for e, r in top_pairs:
        triples = []
        if e in kg and r in kg[e]:
            objs = list(kg[e][r])
            if len(objs) > max_objs_per_rel:
                objs = random.sample(objs, max_objs_per_rel)
            for obj in objs:
                if not has_comma(obj):
                    triples.append((str(e), str(r), str(obj)))
        neighbors_ent.append(triples)
        num_ent += len(triples)
    if num_ent == 0:
        return "No"

    flat_for_prompt = [t for group in neighbors_ent for t in group]                      
    if len(flat_for_prompt) <= topk_ents_by_llm:
        return flat_for_prompt
    top_triples = rank_triples_by_llm(client, question, flat_for_prompt, topk_ents_by_llm, max_tokens)
    return top_triples


def get_two_hop_neighbors_rels_ranking(client, question, top_1hop_triplets, kg):
    neighbors_rel = set()                       
    for h, r, t in top_1hop_triplets:
        if t in kg:
            for rel in kg[t].keys():
                if rel != "~" + r and r != "~" + rel:
                    if not has_comma(t) and not has_comma(rel):
                        neighbors_rel.add((str(t), str(rel)))

    if not neighbors_rel:
        return "No"
    rel_pool = list(neighbors_rel)
    if len(rel_pool) > max_rel_candidates:
        rel_pool = random.sample(rel_pool, max_rel_candidates)
    top_pairs = rank_pairs_by_llm(client, question, rel_pool, topk_rel_by_llm, max_tokens)   # 2) 让 LLM 选 Top-5 (entity, relation)
    
    neighbors_ent = []
    for e, r in top_pairs:
        triples  = []
        if e in kg and r in kg[e]:
            objs = list(kg[e][r])
            if len(objs) > max_objs_per_rel:
                objs = random.sample(objs, max_objs_per_rel)
            for obj in objs:
                if not has_comma(obj):
                    triples.append((str(e), str(r), str(obj)))
        neighbors_ent.append(triples)
    all_tris = [t for group in neighbors_ent for t in group]
    if not all_tris:
        return "No"
    if len(all_tris) <= topk_ents_by_llm:
        top_tris = all_tris
    else:
        top_tris = rank_triples_by_llm(client, question, all_tris, topk_ents_by_llm, max_tokens)

    top_neighbors_ent = []
    for (e2, r2, t2) in top_tris:
        for (h1, r1, t1) in top_1hop_triplets:
            if t1 == e2:
                top_neighbors_ent.append((str(h1), str(r1), str(e2), str(r2), str(t2)))
                break

    if top_1hop_triplets:
        h0, r0, h_n = random.choice(top_1hop_triplets)
        if h_n in kg:
            r_n = random.choice(list(kg[h_n].keys()))
            t_n = random.choice(list(kg[h_n][r_n]))
            top_neighbors_ent.append((str(h0), str(r0), str(h_n), str(r_n), str(t_n)))
        if h0 in kg:
            r_n1 = random.choice(list(kg[h0].keys()))
            t_n1 = random.choice(list(kg[h0][r_n1]))
            top_neighbors_ent.append((str(h0), str(r_n1), str(t_n1)))
    return top_neighbors_ent

def rank_pairs_by_llm(client, question, rel_pool, topk_rel_by_llm, max_tokens):
    if not rel_pool:
        return []
    if len(rel_pool) <= topk_rel_by_llm:
        return rel_pool[:topk_rel_by_llm]
    prompt = (
        "Each item is a pair (entity, relation). "
        f"Select the {topk_rel_by_llm} most semantically related pairs to the sentence. "
        "Return exactly as '(E1,R1);(E2,R2);...'.\n"
        f"Sentence: {question}\nPairs: " + ";".join([f"({e},{r})" for e, r in rel_pool])
    )
    mtoks = min(max_tokens, 50 + len(rel_pool) * 10)
    resp = llm(client, prompt, max_tokens=mtoks)
    top_pairs = parse_pairs(resp or "", topk_rel_by_llm)
    if not top_pairs:
        top_pairs = rel_pool[:topk_rel_by_llm]
    return top_pairs

def rank_triples_by_llm(client, question, triples_pool, topk_ents_by_llm, max_tokens):
    if not triples_pool:
        return []
    if len(triples_pool) <= topk_ents_by_llm:
        return triples_pool[:topk_ents_by_llm]
    prompt = (
        "These are triples (head, relation, tail). "
        f"Select the {topk_ents_by_llm} most semantically related triples to the sentence. "
        "Return exactly as '(H1,R1,T1);(H2,R2,T2);...'.\n"
        f"Sentence: {question}\nTriples: "
        + ";".join([f"({h},{r},{t})" for h, r, t in triples_pool])
    )
    mtoks = min(max_tokens, 50 + len(triples_pool) * 10)
    resp = llm(client, prompt, max_tokens=mtoks)
    top_triples = parse_triples(resp or "", topk_ents_by_llm)
    if not top_triples:
        top_triples = triples_pool[:topk_ents_by_llm]
    return top_triples


examples_block = """Verify the following claims. The context contains evidence triples in the form [head, relation, tail].
Choose exactly one of {True, False} and provide one-sentence evidence.

Context 1: [['Ahamad_Kadhim', 'clubs', "Al-Zawra'a SC"]]
Claim 1: Ahmad Kadhim Assad's club is Al-Zawra'a SC.
Answer 1: True, based on the evidence set, Ahmad Kadhim Assad's club is Al-Zawra'a SC.

Context 2: [['Bananaman', 'firstAired', '"1983-10-03"'], ['Bananaman', 'starring', 'Tim_Brooke-Taylor']]
Claim 2: Yeah! I know that a TV show, which starred Tim Brooke-Taylor, first aired on 3rd October 1983!
Answer 2: True, the claim is supported by the evidence since Bananaman refers to the TV show.

Context 3: [['Jamie_Lawrence', 'composer', 'Death_on_a_Factory_Farm'], ['Death_on_a_Factory_Farm', 'director', 'Sarah_Teale']]
Claim 3: Really? Jamie Lawrence is the music composer of the 83 minute 'Death on a Factory Farm' film, directed by Sarah Teale!
Answer 3: False, there is no evidence for the 83 minute length.
"""

def original_query(client, question_list, ground_truth_list):
    prompt = ("Verify these claims. Choose one of {True, False}. Return one answer per line (e.g., 'Answer 1: True').\n\n")
    for i, q in enumerate(question_list, 1):
        prompt += f"Claim {i}: {q}\n"
    prompt += "\n"
    resp = llm(client, prompt, max_tokens)
    pred = parse_true_false_lines(resp or "", len(question_list))   # 'true' / 'false' / 'unknown'
    original_correct_list = []
    q_choose_list = []
    for i, (pred_tf, gts) in enumerate(zip(pred, ground_truth_list)):
        gold_true = any(bool(x) for x in gts)
        if pred_tf == "unknown":
            original_correct_list.append(False)
            q_choose_list.append(i)
            continue
        pred_bool = (pred_tf == "true")
        ok = (pred_bool == gold_true)
        original_correct_list.append(ok)
        if not ok:
            q_choose_list.append(i)
    return q_choose_list, original_correct_list

def triples_to_context_str(triplet):    
    if len(triplet) == 3:
        h, r, t = triplet
        if str(r).startswith("~"):
            return f"[[{t}, {r[1:]}, {h}]]"
        return f"[[{h}, {r}, {t}]]"

    elif len(triplet) == 5:
        h1, r1, h2, r2, t2 = triplet
        parts = []
        if str(r1).startswith("~"):
            parts.append(f"[{h2}, {r1[1:]}, {h1}]")
        else:
            parts.append(f"[{h1}, {r1}, {h2}]")
        if str(r2).startswith("~"):
            parts.append(f"[{t2}, {r2[1:]}, {h2}]")
        else:
            parts.append(f"[{h2}, {r2}, {t2}]")
        return "[" + ", ".join(parts) + "]"
    else:
        return "[]"

def context_query(client, question_list, ground_truth_list, subgraph_list, i, q_choose_list, already_pos_list, already_neg_list, pos_sam_list, neg_sam_list):
    question_list2 = [question_list[j] for j in q_choose_list]
    ground_truth_list2 = [ground_truth_list[j] for j in q_choose_list]
    subgraph_list2 = [subgraph_list[j] for j in q_choose_list]

    context_texts = []
    picked_tris = []
    for subgraph in subgraph_list2:
        if i >= len(subgraph):
            context_texts.append("[]")
            picked_tris.append(())
            continue
        tri = subgraph[i]
        picked_tris.append(tri)
        context_texts.append(triples_to_context_str(tri))

    prompt = examples_block
    prompt += f"Now verify the following {len(question_list2)} claims in the same way.\n"
    for j, q in enumerate(question_list2, 1):
        prompt += f"Context {j}: {context_texts[j-1]} Claim {j}: {q}\n"
    prompt += f"Answer 1: " 
    print('prompt:', prompt)
    resp = llm(client, prompt, max_tokens=min(max_tokens, 100 + 40 * len(question_list2)))
    print('resp:', resp)
    pred = parse_true_false_lines(resp or "", len(question_list2))

    context_correct_list = []
    for pred_tf, gts in zip(pred, ground_truth_list2):
        gold_true = any(bool(x) for x in gts)
        if pred_tf == "unknown":
            context_correct_list.append(False)
        else:
            context_correct_list.append((pred_tf == "true") == gold_true)

    for j, ok in enumerate(context_correct_list):
        global_idx = q_choose_list[j]
        tri = picked_tris[j]
        if not tri:
            continue
        if ok and not already_pos_list[global_idx]:
            already_pos_list[global_idx] = True
            pos_sam_list[global_idx] = [tri, context_texts[j]]
        if (not ok) and not already_neg_list[global_idx]:
            already_neg_list[global_idx] = True
            neg_sam_list[global_idx] = [tri, context_texts[j]]

    new_q_choose_list = []
    for j, global_idx in enumerate(q_choose_list):
        if (already_pos_list[global_idx] and already_neg_list[global_idx]):
            continue
        if i + 1 < len(subgraph_list2[j]):
            new_q_choose_list.append(global_idx)
    return new_q_choose_list, context_correct_list, already_pos_list, already_neg_list, pos_sam_list, neg_sam_list


def main():
    parser = argparse.ArgumentParser(description="Parsing input arguments.")
    parser.add_argument("--setting", type=str, required=True, choices=["train", "dev"])
    args = parser.parse_args()
    setting = args.setting

    if setting == "train":
        extracted = "./data/factkg_train_set.jsonl"
        output_file = "output_train.jsonl"
    else:
        extracted = "./data/factkg_dev_set.jsonl"
        output_file = "output_dev.jsonl"

    with open("./data/dbpedia_2015_undirected_light.pickle", "rb") as f:
        kg = pickle.load(f)

    questions_dict = {}
    entity_set_dict = {}
    label_set_dict = {}
    with open(extracted, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            qid = int(data["question_id"])
            questions_dict[qid] = data["question"]
            entity_set_dict[qid] = data["entity_set"]
            label_set_dict[qid] = data["Label"]

    # OpenAI client
    client = OpenAI(
        api_key=open_file("./openai_api_key.txt"),
        base_url="XXX",
    )

    dataset_len = len(questions_dict)
    data_num = list(range(1, dataset_len + 1, 10)) + list(range(2, dataset_len + 1, 10))

    question_id_list: List[int] = []
    question_list: List[str] = []
    entity_set_list: List[List[str]] = []
    ground_truth_list: List[List[bool]] = []
    subgraph_list: List[List[Tuple]] = []
    num_triplets_to_test_list: List[int] = []

    questions_num_in_list = 0

    with open(output_file, "a", encoding="utf-8") as fout:
        for ii in tqdm(data_num):
            if ii not in questions_dict:
                continue

            question = questions_dict[ii]
            entity_set = entity_set_dict[ii]
            ground_truth = label_set_dict[ii]

            subgraph = build_subgraph_rels_ranking(client, question, entity_set, kg)
            if subgraph == "No":
                continue

            num_triplets_to_test = len(subgraph)
            if num_triplets_to_test == 0:
                continue

            questions_num_in_list += 1
            question_id_list.append(ii)
            question_list.append(question)
            entity_set_list.append(entity_set)
            ground_truth_list.append(ground_truth)
            subgraph_list.append(subgraph)
            num_triplets_to_test_list.append(num_triplets_to_test)

            if questions_num_in_list >= 1:
                q_choose_list, original_correct_list = original_query(client, question_list, ground_truth_list)
                if isinstance(q_choose_list, str):
                    question_id_list = []
                    question_list = []
                    entity_set_list = []
                    ground_truth_list = []
                    subgraph_list = []
                    num_triplets_to_test_list = []
                    questions_num_in_list = 0
                    continue

                already_pos_list = [False] * len(question_list)
                already_neg_list = [False] * len(question_list)
                pos_sam_list: List[Any] = [False] * len(question_list)
                neg_sam_list: List[Any] = [False] * len(question_list)

                for i in range(max(num_triplets_to_test_list)):
                    if len(q_choose_list) > 0:
                        q_choose_list, context_correct_list, already_pos_list, already_neg_list, pos_sam_list, neg_sam_list = context_query(
                            client,
                            question_list,
                            ground_truth_list,
                            subgraph_list,
                            i,
                            q_choose_list,
                            already_pos_list,
                            already_neg_list,
                            pos_sam_list,
                            neg_sam_list,
                        )
                    if isinstance(q_choose_list, str):
                        break
                for i in range(len(question_list)):
                    if already_pos_list[i] and already_neg_list[i]:
                        result_dict = {
                            "question_id": question_id_list[i],
                            "question": question_list[i],
                            "entity_set": entity_set_list[i],
                            "ground_truth": ground_truth_list[i],
                            "pos_triplet": pos_sam_list[i][0],
                            "pos_context": pos_sam_list[i][1],
                            "neg_triplet": neg_sam_list[i][0],
                            "neg_context": neg_sam_list[i][1],
                        }
                        fout.write(json.dumps(result_dict) + "\n")
                question_id_list = []
                question_list = []
                entity_set_list = []
                ground_truth_list = []
                subgraph_list = []
                num_triplets_to_test_list = []
                questions_num_in_list = 0


if __name__ == "__main__":
    main()

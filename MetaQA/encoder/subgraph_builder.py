import json
import re
import pickle
import argparse
import random
from typing import Sequence, Tuple, Dict
from tqdm import tqdm
from util import open_file, make_openai_client, llm, KGUtils

parser = argparse.ArgumentParser(description="Parsing input arguments.")
parser.add_argument('--setting', type=str, required=True, choices=['train', 'dev'], help="Dataset type")
parser.add_argument('--hop', type=int, required=True, choices=[1, 2, 3], help="Number of hops")
args = parser.parse_args()
hop = args.hop
setting = args.setting

file_map = {
    'train': {
        1: ('../data/1_hop_train.jsonl', './dataset/1-hop-trainset.jsonl'),
        2: ('../data/2_hop_train.jsonl', './dataset/2-hop-trainset.jsonl'),
        3: ('../data/3_hop_train.jsonl', './dataset/3-hop-trainset.jsonl')
    },
    'dev': {
        1: ('../data/1_hop_dev.jsonl', './dataset/1-hop-devset.jsonl'),
        2: ('../data/2_hop_dev.jsonl', './dataset/2-hop-devset.jsonl'),
        3: ('../data/3_hop_dev.jsonl', './dataset/3-hop-devset.jsonl')
    }
}
data_path, save_file = file_map[setting][hop]

with open('../data/metaqa_kg.pickle', 'rb') as f:
    knowledge_graph = pickle.load(f)

questions_dict, entity_set_dict, label_set_dict = {}, {}, {}
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        dataset = json.loads(line)
        qid = dataset["question_id"]
        questions_dict[qid] = dataset["question"]
        entity_set_dict[qid] = dataset["entity_set"]
        label_set_dict[qid] = dataset["Label"]

no_evidence_prompt = open_file(f'../prompts/meta_{hop}_hop_prompts/verify_claim_no_evidence.txt')
context_prompt = open_file(f'../prompts/meta_{hop}_hop_prompts/verify_claim_with_evidence.txt')
api_key = open_file('../openai_api_key.txt')
kb = '../data/kb.txt'
base_url = "xxxxx"       # input your url
client = make_openai_client(api_key, base_url=base_url)
model_name = 'gpt-4o-mini'
max_tokens = 500
llm_func = lambda prompt: llm(client, prompt, model_name=model_name, max_tokens=max_tokens, temperature=0.2, top_p=0.1, timeout=30, retries=5)
u: KGUtils = None  


def get_one_hop_neighbors(question, entity_set, knowledge_graph, top_k):
    relations = set()
    for entity in entity_set:
        if entity not in knowledge_graph:
            continue
        for relation in knowledge_graph[entity].keys():
            if (',' not in str(entity)) and (',' not in str(relation)):
                relations.add((str(entity), str(relation)))
    if not relations:
        return "No"
    ranked_relations = KGUtils.get_top_related_triplets(question, relations, top_k)
    if ranked_relations == "No" or not ranked_relations:
        return "No"
    triplets = []
    valid_relations = [rel for rel in ranked_relations if len(rel) == 2]
    for subject, relation in valid_relations:
        if subject in knowledge_graph and relation in knowledge_graph[subject]:
            for obj in knowledge_graph[subject][relation]:
                if ',' not in str(obj):
                    triplets.append((subject, relation, str(obj)))
    if not triplets:
        return "No"
    ranked_triplets = KGUtils.get_top_entity_triplets(question, triplets, len(triplets), top_k)
    return ranked_triplets if ranked_triplets else "No"


def get_two_hop_neighbors(question, top_1hop_triplets, knowledge_graph, top_k):
    relations = set()
    for triplet in top_1hop_triplets:
        try:
            h, r, t = triplet
        except Exception:
            return "No"
        if t in knowledge_graph:
            for relation in knowledge_graph[t].keys():
                if relation != '~' + r and r != '~' + relation:
                    if (str(t), str(relation)) not in relations and (',' not in str(t)) and (',' not in str(relation)):
                        relations.add((str(t), str(relation)))
    top_neighbors_rel = KGUtils.get_top_related_triplets(question, relations, top_k)
    if top_neighbors_rel == "No":
        return "No"
    neighbors_ent = [[] for _ in range(len(top_neighbors_rel))]
    for i, (head, relation) in enumerate(top_neighbors_rel):
        if head in knowledge_graph and relation in knowledge_graph[head]:
            for obj in knowledge_graph[head][relation]:
                if (str(head), str(relation), str(obj)) not in neighbors_ent[i] and (',' not in str(obj)):
                    neighbors_ent[i].append((str(head), str(relation), str(obj)))
    top_neighbors_ent = KGUtils.get_top_entity_triplets(question, neighbors_ent, len(neighbors_ent), top_k)
    if top_neighbors_ent == "No":
        return "No"
    flat = [item for sub in top_neighbors_ent for item in sub]
    if not flat:
        return "No"
    for i in range(len(flat)):
        for triplet in top_1hop_triplets:
            h, r, t = triplet
            if t == flat[i][0]:
                flat[i] = (str(h), str(r), str(flat[i][0]), str(flat[i][1]), str(flat[i][2]))
                break
    if top_1hop_triplets:
        h0, r0, h_n = random.choice(list(top_1hop_triplets))
        if h_n in knowledge_graph and knowledge_graph[h_n]:
            r_n = random.choice(list(knowledge_graph[h_n]))
            t_n = random.choice(list(knowledge_graph[h_n][r_n]))
            flat.append((str(h0), str(r0), str(h_n), str(r_n), str(t_n)))
    return flat


def get_three_hop_neighbors(question, top_1hop_triplets, knowledge_graph, top_k):
    def is_inverse(a, b):
        return (isinstance(a, str) and isinstance(b, str)) and (a == '~' + b or b == '~' + a)
    def bad_atom(x):
        return ',' in str(x)
    if top_1hop_triplets in ["No", None] or len(top_1hop_triplets) == 0:
        return "No"
    top_2hop_triplets = get_two_hop_neighbors(question, top_1hop_triplets, knowledge_graph, top_k)
    if top_2hop_triplets == "No" or not top_2hop_triplets:
        return "No"
    relations = set()
    for tup in top_2hop_triplets:
        try:
            h0, r0, h1, r1, t1 = tup
        except Exception:
            return "No"
        if t1 in knowledge_graph:
            for r2 in knowledge_graph[t1].keys():
                if is_inverse(r2, r1):
                    continue
                if bad_atom(t1) or bad_atom(r2):
                    continue
                relations.add((str(t1), str(r2)))
    top_neighbors_rel = KGUtils.get_top_related_triplets(question, relations, top_k)
    if top_neighbors_rel == "No" or not top_neighbors_rel:
        return "No"
    neighbors_ent = [[] for _ in range(len(top_neighbors_rel))]
    for i, (head, relation) in enumerate(top_neighbors_rel):
        if head in knowledge_graph and relation in knowledge_graph[head]:
            for obj in knowledge_graph[head][relation]:
                if not bad_atom(obj):
                    trip = (str(head), str(relation), str(obj))
                    if trip not in neighbors_ent[i]:
                        neighbors_ent[i].append(trip)
    if all(len(g) == 0 for g in neighbors_ent):
        return "No"
    top_neighbors_ent = KGUtils.get_top_entity_triplets(question, neighbors_ent, len(neighbors_ent), top_k)
    if top_neighbors_ent == "No" or not top_neighbors_ent:
        return "No"
    flat_third = []
    if isinstance(top_neighbors_ent, list) and top_neighbors_ent and isinstance(top_neighbors_ent[0], list):
        for g in top_neighbors_ent:
            flat_third.extend(g)
    else:
        flat_third = list(top_neighbors_ent)
    results, seen = [], set()
    def push(cand):
        if any(bad_atom(x) for x in cand):
            return
        if cand in seen:
            return
        seen.add(cand)
        results.append(cand)

    for (h2, r2, t2) in flat_third:
        for (h0, r0, h1, r1, t1) in top_2hop_triplets:
            if t1 == h2 and not (r2 == '~' + r1 or r1 == '~' + r2):
                push((str(h0), str(r0), str(h1), str(r1), str(h2), str(r2), str(t2)))
                break
    try:
        if top_2hop_triplets:
            h0, r0, h1, r1, t1 = random.choice(list(top_2hop_triplets))
            if t1 in knowledge_graph and knowledge_graph[t1]:
                for _ in range(5):
                    r_rand = random.choice(list(knowledge_graph[t1]))
                    if is_inverse(r_rand, r1):
                        continue
                    t_rand = random.choice(list(knowledge_graph[t1][r_rand]))
                    cand = (str(h0), str(r0), str(h1), str(r1), str(t1), str(r_rand), str(t_rand))
                    if not any(bad_atom(x) for x in cand):
                        push(cand)
                        break
    except Exception:
        pass
    return results if results else "No"

def build_subgraph(question, entity_set, knowledge_graph, top_k, hop):
    one_hop = get_one_hop_neighbors(question, entity_set, knowledge_graph, top_k)         # 1-hop
    if one_hop == "No":
        return "No"
    if hop == 1:
        return one_hop

    two_hop = get_two_hop_neighbors(question, one_hop, knowledge_graph, top_k)           # 2-hop
    if two_hop == "No":
        return "No"
    if hop == 2:
        return two_hop
    
    three_hop = get_three_hop_neighbors(question, one_hop, knowledge_graph, top_k)       # 3-hop
    if three_hop == "No":
        return "No"
    three_hop = KGUtils.filter_and_dedup_candidates(three_hop, forbid_meta=True)
    if three_hop == "No":
        return "No"
    return three_hop


def context_query(question, ground_truth, triplet, context_prompt, llm_func, positive_sample):
    def format_subtriplet(e1, rel, e2):
        if isinstance(rel, str):
            return f"[{e2}, {rel[1:]}, {e1}]" if rel.startswith('~') else f"[{e1}, {rel}, {e2}]"
        else:
            return None

    pieces = []
    if len(triplet) == 3:
        pieces.append(format_subtriplet(triplet[0], triplet[1], triplet[2]))
    elif len(triplet) == 5:
        pieces.append(format_subtriplet(triplet[0], triplet[1], triplet[2]))
        pieces.append(format_subtriplet(triplet[2], triplet[3], triplet[4]))
    elif len(triplet) == 7:
        pieces.append(format_subtriplet(triplet[0], triplet[1], triplet[2]))
        pieces.append(format_subtriplet(triplet[2], triplet[3], triplet[4]))
        pieces.append(format_subtriplet(triplet[4], triplet[5], triplet[6]))
    else:
        return 'triplet error', positive_sample
    if any(p is None for p in pieces):
        return 'triplet error', positive_sample

    context = "[" + ", ".join(pieces) + "]"
    result = llm_func(context_prompt.replace('<<<<CLAIM>>>>', question).replace('<<<<EVIDENCE_SET>>>>', context))
    # print('result:', result)
    if result is None:
        return 'triplet error', positive_sample
    ans = KGUtils.extract_final_answer(result) 
    if ans is None:
        return 'triplet error', positive_sample
    parts = re.split(r'\s*(?:,|/|and|or)\s*', ans)
    gt = {str(x).strip().lower() for x in ground_truth}
    hit_gt = any(KGUtils.norm(p) in gt for p in parts if p.strip())

    rel_tail_rel, rel_tail_obj = KGUtils.candidate_tail_from_path(triplet)
    hit_tail = False
    if rel_tail_obj is not None:
        cand = KGUtils.norm(rel_tail_obj)
        hit_tail = any(KGUtils.norm(p) == cand for p in parts if p.strip())
    context_correct = hit_gt and hit_tail
    if context_correct:
        positive_sample = triplet
    return context_correct, positive_sample


def process_question(idx, question, entity, ground_truth, unique_relations, knowledge_graph, top_k, hop, no_evidence_prompt, context_prompt, llm_func):
    llm_result = KGUtils.no_evidence_query(question, ground_truth, no_evidence_prompt, llm_func)
    if llm_result is True or llm_result is None:
        return None
    if isinstance(llm_result, str) and llm_result.startswith("Answer:"):
        llm_result = llm_result[len("Answer:"):].strip()

    subgraph = build_subgraph(question, entity, knowledge_graph, top_k, hop)
    if subgraph in ["No", None] or len(subgraph) == 0:
        return None
    positive_sample_list = []
    positive_sample = False
    for triple in subgraph:
        context_correct, positive_sample = context_query(question, ground_truth, triple, context_prompt, llm_func, positive_sample)
        if context_correct == 'triplet error':
            continue
        if context_correct is True and positive_sample and positive_sample not in positive_sample_list:
            positive_sample_list.append(positive_sample)
    if not positive_sample_list:
        return None
    sample_to_use = positive_sample_list[0]
    if not isinstance(sample_to_use, (list, tuple)):
        return None

    if len(sample_to_use) == 3:                               # 1-hop: (h0, r0, h0)
        h0, r0, t0 = sample_to_use
        neg_tail = u.pick_negative_tail(h0, r0, t0, knowledge_graph=knowledge_graph, ground_truth=ground_truth)
        if not neg_tail:
            return None
        random_relation = u.pick_relation(r0)
        negative_triplet = [str(h0), str(random_relation[0]), str(neg_tail)]
    
    elif len(sample_to_use) == 5:                             # 2-hop: (h0, r0, h1, r1, t1)
        h0, r0, h1, r1, t1 = sample_to_use
        neg_tail = u.pick_negative_tail(h1, r1, t1, ground_truth=ground_truth)
        if not neg_tail:
            return None
        random_relation = u.pick_relation(r1)
        negative_triplet = [str(h0), str(r0), str(h1), str(random_relation[0]), str(neg_tail)]

    elif len(sample_to_use) == 7:                             # 3-hop: (h0, r0, h1, r1, h2, r2, t2)
        h0, r0, h1, r1, h2, r2, t2 = sample_to_use
        neg_tail = u.pick_negative_tail(h2, r2, t2, ground_truth=ground_truth)
        if not neg_tail:
            return None
        random_relation2 = u.pick_relation(r1)
        random_relation3 = u.pick_relation(r2)
        negative_triplet = [str(h0), str(r0), str(h1), str(random_relation2[0]), str(h2), str(random_relation3[0]), str(neg_tail)]

    else:
        print(f"[ERROR] Unsupported positive sample length: {len(sample_to_use)}")
        return None
    return {
        'question_id': idx,
        'question': question,
        'ground_truth': ground_truth,
        'positive_triplets': positive_sample_list,
        'negative_triplet': negative_triplet,
    }


def main():
    print(f"Start building %d-hop %s subgraphs!" % (hop, setting))
    top_k = 20
    global u
    unique_relations = KGUtils.extract_unique_relations(kb)
    u = KGUtils(knowledge_graph=knowledge_graph, unique_relations=unique_relations)
    dataset_length = len(questions_dict)
    indices_to_process = range(1, dataset_length + 1)
    with open(save_file, 'a', encoding='utf-8') as f:
        for i in tqdm(indices_to_process):
            # if i % 5 != 0:
            #     continue
            question = questions_dict[i]
            entity = entity_set_dict[i]
            ground_truth = label_set_dict[i]

            result = process_question(idx=i, question=question, entity=entity, ground_truth=ground_truth, unique_relations=unique_relations,
                knowledge_graph=knowledge_graph, top_k=top_k, hop=hop, no_evidence_prompt=no_evidence_prompt, context_prompt=context_prompt,
                llm_func=llm_func)
            if result:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()

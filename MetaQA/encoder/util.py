import json
import re
import time
import random
from functools import lru_cache
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Optional, Set, Dict
from openai import OpenAI

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def make_openai_client(api_key, base_url=None):
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)

def llm(client: OpenAI, prompt, *, model_name, max_tokens, temperature, top_p, timeout, retries):
    for attempt in range(retries):
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
                timeout=timeout,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                print("[ERROR]", e)
            time.sleep(5)
    return None


class KGUtils:
    token_re = re.compile(r"[A-Za-z0-9_]+")
    def __init__(self, knowledge_graph, unique_relations):
        self.knowledge_graph = knowledge_graph or {}
        self.unique_relations = unique_relations or set()

    @staticmethod
    @lru_cache(maxsize=4096)
    def _tok(s):
        if s is None:
            return tuple()
        return tuple(KGUtils.token_re.findall(str(s).lower()))

    @staticmethod
    def tuple_text(t) :
        return " ".join(map(str, t))

    @staticmethod
    def extract_unique_relations(kb_path):
        relations = set()
        with open(kb_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 3:
                    relations.add(parts[1])
        return relations

    @staticmethod
    def extract_final_answer(text):
        if not text:
            return None
        s = text.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "answer" in obj:
                return str(obj["answer"]).strip().strip('"').strip("'")
        except Exception:
            pass
        for line in reversed(s.splitlines()):
            m = re.search(r'(?i)\b(?:final answer|answer)\s*[:：]\s*(.+?)\s*$', line.strip())
            if m:
                return m.group(1).strip().strip('"').strip("'")
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", s)
        if quoted:
            return quoted[-1].strip()
        years = re.findall(r'\b(?:18|19|20)\d{2}\b', s)
        if years:
            return years[-1]
        return s.splitlines()[-1].strip()

    @staticmethod
    def rel_base(r):
        return (r or "").lstrip("~").strip()

    @staticmethod
    def is_inverse_pair(r1, r2):
        return KGUtils.rel_base(r1) == KGUtils.rel_base(r2) and ((r1 or "").startswith("~") != (r2 or "").startswith("~"))

    @staticmethod
    def relations_in_path(triplet):
        rels = []
        if len(triplet) >= 3: rels.append(str(triplet[1]))
        if len(triplet) >= 5: rels.append(str(triplet[3]))
        if len(triplet) >= 7: rels.append(str(triplet[5]))
        return rels

    @staticmethod
    def entities_in_path(triplet):
        ents: List[str] = []
        if len(triplet) >= 3:
            ents = [str(triplet[0]), str(triplet[2])]
        if len(triplet) >= 5:
            ents.append(str(triplet[4]))
        if len(triplet) >= 7:
            ents.append(str(triplet[6]))
        return ents

    @classmethod
    def has_immediate_backtrack(cls, triplet):
        rels = cls.relations_in_path(triplet)
        ents = cls.entities_in_path(triplet)
        for i in range(len(rels) - 1):
            if cls.is_inverse_pair(rels[i], rels[i+1]):
                if i + 2 < len(ents) and ents[i] == ents[i+2]:
                    return True
        return False

    @classmethod
    def reduce_inverse_backtracks(cls, rels, ents):
        i = 0
        out_rels: List[str] = []
        out_ents: List[str] = [ents[0]] if ents else []
        while i < len(rels):
            if i + 1 < len(rels) and cls.is_inverse_pair(rels[i], rels[i+1]):
                if i + 2 < len(ents) and ents[i] == ents[i+2]:
                    i += 2
                    continue
            out_rels.append(rels[i])
            if i + 1 < len(ents):
                out_ents.append(ents[i+1])
            i += 1
        return out_rels, out_ents

    @classmethod
    def canonicalize_triplet_path(cls, triplet):
        rels = cls.relations_in_path(triplet)
        ents = cls.entities_in_path(triplet)
        r2, e2 = cls.reduce_inverse_backtracks(rels, ents)
        flat: List[str] = []
        if not e2:
            return tuple()
        flat.append(e2[0])
        for i, r in enumerate(r2):
            flat.extend([r, e2[i+1]])
        return tuple(flat)

    @classmethod
    def iter_flat_tuples(cls, neighbors_ent):
        out: List[Tuple[str, ...]] = []
        if neighbors_ent is None:
            return out
        base = list(neighbors_ent) if not isinstance(neighbors_ent, list) else neighbors_ent
        for item in base:
            if isinstance(item, tuple) and 2 <= len(item) <= 3 and all(not isinstance(x, (list, tuple, set)) for x in item):
                out.append(tuple(map(str, item)))
            elif isinstance(item, (list, set, tuple)):
                for sub in list(item):
                    if isinstance(sub, tuple) and 2 <= len(sub) <= 3 and all(not isinstance(x, (list, tuple, set)) for x in sub):
                        out.append(tuple(map(str, sub)))
        return out

    @classmethod
    def score_tuple(cls, question_tokens_set, question_text, tup):
        h = tup[0] if len(tup) > 0 else ""
        r = tup[1] if len(tup) > 1 else ""
        t = tup[2] if len(tup) > 2 else ""
        ht = set(cls._tok(h))
        rt = set(cls._tok(r))
        tt = set(cls._tok(t))
        overlap = len(question_tokens_set & ht) + 1.5 * len(question_tokens_set & rt) + len(question_tokens_set & tt)
        bonus = 0.5 if (str(r).lower() and str(r).lower() in question_text) else 0.0
        length_penalty = 0.01 * len(cls.tuple_text(tup))
        return overlap + bonus - length_penalty

    @classmethod
    def stable_sort_by_score(cls, cands, score_map):
        return sorted(cands, key=lambda x: (-score_map[x], len(cls.tuple_text(x)), cls.tuple_text(x)))

    @classmethod
    def dedup_preserve_order(cls, tuples):
        seen, out = set(), []
        for t in tuples:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    @classmethod
    def get_top_related_triplets(cls, question, neighbors_ent, top_k):
        flat = cls.iter_flat_tuples(neighbors_ent)
        if not flat:
            return "No"
        flat = cls.dedup_preserve_order(flat)
        if len(flat) <= top_k:
            return flat
        q_tokens = set(cls._tok(question))
        q_text = " ".join(cls._tok(question))
        score_map = {t: cls.score_tuple(q_tokens, q_text, t) for t in flat}
        ranked = cls.stable_sort_by_score(flat, score_map)[:top_k]
        return ranked if ranked else "No"

    @classmethod
    def get_top_entity_triplets(cls, question, neighbors_ent, num_ent, top_k):
        if neighbors_ent is None:
            return "No"
        q_tokens = set(cls._tok(question))
        q_text = " ".join(cls._tok(question))
        is_grouped = isinstance(neighbors_ent, list) and neighbors_ent and isinstance(neighbors_ent[0], list)
        if not is_grouped:
            flat = [tuple(map(str, t)) for t in list(neighbors_ent) if isinstance(t, tuple) and len(t) == 3]
            if not flat:
                return "No"
            if len(flat) <= top_k:
                return flat
            score_map = {t: cls.score_tuple(q_tokens, q_text, t) for t in flat}
            return cls.stable_sort_by_score(flat, score_map)[:top_k]

        groups = neighbors_ent
        scored_groups = []
        total_candidates = 0
        for g in groups:
            gi = [tuple(map(str, t)) for t in g if isinstance(t, tuple) and len(t) == 3]
            total_candidates += len(gi)
            if not gi:
                scored_groups.append([])
                continue
            score_map = {t: cls.score_tuple(q_tokens, q_text, t) for t in gi}
            scored_groups.append(cls.stable_sort_by_score(gi, score_map))
        if total_candidates == 0:
            return "No"
        if total_candidates <= top_k:
            return [g[:2] for g in scored_groups]
        picks_per_group = [0] * len(scored_groups)
        result_groups = [[] for _ in range(len(scored_groups))]
        picked = 0
        for _round in range(2):
            if picked >= top_k:
                break
            for gi, g in enumerate(scored_groups):
                if picked >= top_k:
                    break
                if picks_per_group[gi] >= 2:
                    continue
                if picks_per_group[gi] < len(g):
                    result_groups[gi].append(g[picks_per_group[gi]])
                    picks_per_group[gi] += 1
                    picked += 1
        return result_groups if any(result_groups) else "No"

    @classmethod
    def filter_and_dedup_candidates(cls, cands, forbid_meta):
        kept = []
        seen = set()
        for t in cands:
            if cls.has_immediate_backtrack(t):
                continue
            canon = cls.canonicalize_triplet_path(t)
            if not canon:
                continue
            if canon in seen:
                continue
            seen.add(canon)
            kept.append(t)
        return kept

    @staticmethod
    def is_year(s):
        return bool(re.fullmatch(r'(18|19|20)\d{2}', str(s).strip()))

    @staticmethod
    def norm(s):
        return str(s).strip().strip('"').strip("'").lower()

    @staticmethod
    def candidate_tail_from_path(triplet):
        if not isinstance(triplet, tuple):
            return None, None
        if len(triplet) == 3:
            return triplet[1], triplet[2]
        if len(triplet) == 5:
            return triplet[3], triplet[4]
        if len(triplet) == 7:
            return triplet[5], triplet[6]
        return None, None

    def pick_negative_tail(self, head, relation, positive_tail, knowledge_graph=None, ground_truth=None):
        kg = knowledge_graph if knowledge_graph is not None else self.knowledge_graph or {}
        head = str(head).strip()
        relation = str(relation).strip()
        base_rel = relation[1:] if relation.startswith('~') else relation
        positive_tail = str(positive_tail).strip()
        gt_set = {str(x).strip().lower() for x in (ground_truth or [])}

        def ok(c):
            c = str(c).strip()
            return c and c.lower() not in gt_set and c != positive_tail

        if head in kg and base_rel in kg[head]:
            pool = [str(o).strip() for o in kg[head][base_rel] if ok(o)]
            if pool:
                return random.choice(pool)

        if base_rel == 'release_year' or self.is_year(positive_tail):
            for _ in range(30):
                y = str(random.randint(1950, 2015))
                if ok(y):
                    return y

        for h2, rels in kg.items():
            if base_rel in rels:
                for cand in rels[base_rel]:
                    if ok(cand):
                        return str(cand).strip()

        if head in kg:
            for r_alt, objs in kg[head].items():
                if r_alt == base_rel:
                    continue
                for cand in objs:
                    if ok(cand):
                        return str(cand).strip()

        return None

    def pick_relation(self, relations, unique_relations: Optional[Set[str]] = None):
        select_relation = []
        uniq = unique_relations if unique_relations is not None else self.unique_relations
        if relations.startswith('~'):
            relation = relations[1:]
            candidates = [rel for rel in uniq if rel != relation]
            if candidates:
                select_relation.append('~' + random.choice(candidates))
        else:
            relation = relations
            candidates = [rel for rel in uniq if rel != relation]
            if candidates:
                select_relation.append(random.choice(candidates))
        return select_relation


    @staticmethod
    def no_evidence_query(question, ground_truth, no_evidence_prompt, llm_func):
        prompt = no_evidence_prompt.replace('<<<<CLAIM>>>>', question)
        result = llm_func(prompt)
        if result is None or result.strip() == '{}':
            return None
        ans = KGUtils.extract_final_answer(result)
        if ans:
            parts = re.split(r'\s*(?:,|/|and|or)\s*', ans)
            gt = {str(x).strip().lower() for x in ground_truth}
            if any(p.strip().lower() in gt for p in parts if p.strip()):
                return True
        return result

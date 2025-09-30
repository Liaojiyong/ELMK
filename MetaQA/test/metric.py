from __future__ import annotations
from datetime import timedelta 
from typing import Any, List, Sequence, Tuple

__all__ = [
    "normalize_answer_list",
    "compute_metrics",
    "log_metrics_to_file",
    "MetricsTracker",
]

def normalize_answer_list(obj):
    if obj is None:
        return []
    if isinstance(obj, (list, tuple, set)):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        return [str(v) for v in obj.values()]
    s = str(obj).strip()
    if not s:
        return []
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines if lines else [s]

def compute_metrics(answer_list, ground_truth_list,):
    if not isinstance(answer_list, (list, tuple, set)):
        answer_list = [answer_list] if answer_list is not None else []
    answer_list = list(answer_list)
    ground_truth = ground_truth_list[0]
    ground_truth_cleaned = [item.strip().lower() for item in ground_truth]
    answer_set = set(str(item).strip().lower() for item in answer_list)
    ground_truth_set = set(ground_truth_cleaned)
    correct_matches = answer_set.intersection(ground_truth_set)
    precision = len(correct_matches) / len(answer_set) if len(answer_set) > 0 else 0.0
    recall = len(correct_matches) / len(ground_truth_set) if len(ground_truth_set) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    context_correct_list = [False] * 1
    if correct_matches:
        context_correct_list[0] = True
    # print('context_correct_list:', context_correct_list)
    return precision, recall, f1, context_correct_list

def log_metrics_to_file(
    step_tag: str,
    processed: int,
    elapsed_seconds: float,
    total_correct: int,
    total_f1: float,
    total_precision_sum: float,
    total_recall_sum: float,
    sum_prompt_tokens: float,
    sum_completion_tokens: float,
    sum_total_tokens: float,
    answered_questions: int,
    log_file: str = "metrics_log.txt",
):
    if processed <= 0:
        return
    acc = total_correct / processed
    precision_mean = total_precision_sum / processed
    recall_mean = total_recall_sum / processed
    f1_mean = total_f1 / processed
    if answered_questions > 0:
        avg_prompt_tok = sum_prompt_tokens / answered_questions
        avg_completion_tok = sum_completion_tokens / answered_questions
        avg_total_tok = sum_total_tokens / answered_questions
    else:
        avg_prompt_tok = avg_completion_tok = avg_total_tok = 0.0
    avg_time_per_q = elapsed_seconds / processed
    human_elapsed = str(timedelta(seconds=int(elapsed_seconds)))
    lines = [
        f"[{step_tag}] Processed: {processed}",
        f"ACC: {total_correct}/{processed} = {acc:.6f}",
        f"Precision (mean over questions): {precision_mean:.6f}",
        f"Recall    (mean over questions): {recall_mean:.6f}",
        f"F1        (mean over questions): {f1_mean:.6f}",
        (
            "Avg tokens per question — "
            f"prompt: {avg_prompt_tok:.2f}, completion: {avg_completion_tok:.2f}, total: {avg_total_tok:.2f}"
        ),
        (
            f"Elapsed: {elapsed_seconds:.2f}s ({human_elapsed}) | "
            f"Avg per question: {avg_time_per_q:.4f}s"
        ),
        "-" * 60,
    ]
    text = "\n".join(lines)
    print("\n" + text + "\n")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")

class MetricsTracker:
    def __init__(self, log_file, print_each_batch):
        self.log_file = log_file
        self.print_each_batch = print_each_batch
        self.total_correct = 0
        self.total_f1 = 0.0
        self.total_precision_sum = 0.0
        self.total_recall_sum = 0.0
        self.sum_prompt_tokens = 0.0
        self.sum_completion_tokens = 0.0
        self.sum_total_tokens = 0.0
        self.answered_questions = 0
        self.processed = 0

    def on_batch(self, question_list, ground_truth_list, result_obj, usage_tuple, context_texts=None):
        answer_list = normalize_answer_list(result_obj)
        precision, recall, f1, context_correct_list = compute_metrics(answer_list, ground_truth_list)

        n_q = max(1, len(question_list))
        pt, ct, tt = usage_tuple
        avg_prompt_tok = pt / n_q
        avg_completion_tok = ct / n_q
        avg_total_tok = tt / n_q
        self.processed += 1
        self.total_precision_sum += precision
        self.total_recall_sum += recall
        self.total_f1 += f1
        self.total_correct += context_correct_list.count(True)
        self.sum_prompt_tokens += avg_prompt_tok
        self.sum_completion_tokens += avg_completion_tok
        self.sum_total_tokens += avg_total_tok
        self.answered_questions += n_q
        if self.print_each_batch:
            lines = []
            lines.extend(
                [
                    f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}",
                    (
                        "Avg tokens per Q — "
                        f"prompt: {avg_prompt_tok:.2f}, "
                        f"completion: {avg_completion_tok:.2f}, "
                        f"total: {avg_total_tok:.2f}"
                    ),
                    "-" * 60,
                ]
            )
            print("\n".join(lines))
        return precision, recall, f1

    def periodic_log(self, step_tag, elapsed_seconds):
        log_metrics_to_file(
            step_tag=step_tag,
            processed=self.processed,
            elapsed_seconds=elapsed_seconds,
            total_correct=self.total_correct,
            total_f1=self.total_f1,
            total_precision_sum=self.total_precision_sum,
            total_recall_sum=self.total_recall_sum,
            sum_prompt_tokens=self.sum_prompt_tokens,
            sum_completion_tokens=self.sum_completion_tokens,
            sum_total_tokens=self.sum_total_tokens,
            answered_questions=self.answered_questions,
            log_file=self.log_file,
        )

    def final_log(self, elapsed_seconds):
        self.periodic_log("Final", elapsed_seconds)
        if self.processed > 0:
            final_avg_precision = self.total_precision_sum / self.processed
            final_avg_recall = self.total_recall_sum / self.processed
            acc = self.total_correct / self.processed
            lines = [
                f"ACC: {self.total_correct}/{self.processed} = {acc:.6f}",
                f"Precision (mean over questions): {final_avg_precision:.6f}",
                f"Recall    (mean over questions): {final_avg_recall:.6f}",
                f"F1: {self.total_f1}/{self.processed} = {self.total_f1 / self.processed:.6f}",
                "-" * 60,
            ]
            text = "\n".join(lines)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(text + "\n")
import pickle
import json
import jsonlines
import argparse
import re
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsing input arguments.")
    parser.add_argument('--hop', nargs='+', default=['1-hop', '2-hop', '3-hop'], help='List of hop counts (e.g., 1-hop 2-hop 3-hop)')  
    parser.add_argument('--kb', type=str, default='./kb.txt', help='Path for metaqa kb.')
    args = parser.parse_args()
    settings = ['train', 'dev', 'test']

    KG_construct = {}
    with open(args.kb, 'r') as f:
        for line in f:
            head, relation, tail = line.strip().split('|')
            try:
                KG_construct[head][relation].append(tail)
            except:
                try:
                    KG_construct[head][relation] = [tail]
                except:
                    KG_construct[head] = {}
                    KG_construct[head][relation] = [tail]

            try:
                KG_construct[tail]['~'+relation].append(head)
            except:
                try:
                    KG_construct[tail]['~'+relation] = [head]
                except:
                    KG_construct[tail] = {}
                    KG_construct[tail]['~'+relation] = [head]

    with open('./metaqa_kg.pickle', 'wb') as f:
        pickle.dump(KG_construct, f)

    def build_dataset(input_file, output_file, dataset_type="test"):
        dataset = {}
        with open(input_file, 'r') as f:
            for line in f:
                seperated = line.strip().split('\t')
                entities = re.findall(r'\[(.*?)\]', seperated[0])
                labels = seperated[1].split('|')
                question = seperated[0] + '?'
                dataset[question] = {
                    'entity_set': [entities[0]], 
                    'Label': labels
                }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with jsonlines.open(output_file, mode='w') as w:
            for i, sample in enumerate(dataset.keys(), start=1):
                new_sample = {
                    "question_id": i,
                    "question": sample,
                    "entity_set": dataset[sample]["entity_set"],
                    "Label": dataset[sample]["Label"],
                }
                w.write(new_sample)
        print(f"Successfully processed: {input_file} -> {output_file}")

    for hop in args.hop:
        for dtype in settings: 
            input_file = os.path.join(hop, "vanilla", f"qa_{dtype}.txt")
            output_file = f'./{hop.replace("-", "_")}_{dtype}.jsonl'
            build_dataset(input_file, output_file, dtype)

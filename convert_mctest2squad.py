import os
import csv
import json
import argparse

from tqdm import tqdm


def load_tsv_file(data_path):
    with open(data_path, 'r') as fin:
        reader = csv.reader(fin, delimiter='\t')
        data = list(reader)
    return data

def clean_question(q):
    if q.startswith("one: "):
        return q[5:]
    if q.startswith("multiple: "):
        return q[10:]

def convert_mctest_to_squad(data_root, data_type, num_story):
    squad_formatted_content = {"data": [], "version": f"mc{num_story}.{data_type}"}
    unique_id = 100000000
    total = 0
    difficulty_set = ["middle", "high"]
    data = load_tsv_file(os.path.join(data_root, f"mc{num_story}.{data_type}.tsv"))
    ans_data = load_tsv_file(os.path.join(data_root, f"mc{num_story}.{data_type}.ans"))
    for datum, ans_datum in tqdm(zip(data, ans_data)):
        qas = []
        question = [datum[i:i+5] for i in range(3, len(datum), 5)]
        for q in question:
            assert len(q) == 5
        for i, q in enumerate(question):
            prefix = 'one' if "one" in q[0] else 'multiple'
            qas.append({
                "answers":[{
                    "answer_start": -1,
                    "text": None,
                    "answer": ans_datum[i],
                }],
                "question": clean_question(q[0]),
                "id": f"{data_type}-{prefix}-{unique_id}",
                "options": q[1:],
            })
            unique_id += 1
        paragraphs = {
            "title": 'dummy',
            "paragraphs": [{
                "context": datum[2].replace("\\newline", "\n"),
                "qas": qas,
            }]
        }
        total += len(question)
        squad_formatted_content["data"].append(paragraphs)
    return squad_formatted_content


def save(data, dir_root, data_type, num_story):
    file_path = os.path.join(dir_root, f"mc{num_story}.{data_type}.json")
    with open(file_path, 'w') as fout:
        print ("Saving {}".format(file_path))
        json.dump(data, fout, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/mctest/data/MCTest')
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--num_story', type=int, default=500, choices=[160, 500])
    args = parser.parse_args()
    
    data = convert_mctest_to_squad(args.data_dir, args.data_type, args.num_story) 
    save(data, args.data_dir, args.data_type, args.num_story)
    
if __name__ == "__main__":
    main()
    

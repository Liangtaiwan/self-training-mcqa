import os
import json
import argparse

from tqdm import tqdm


def convert_race_to_squad(data_root, data_type):
    squad_formatted_content = {"data": [], "version": f"RACE_{data_type}"}
    data = []
    unique_id = 100000000
    examples = []
    difficulty_set = ["middle", "high"]
    for d in difficulty_set:
        data_path = os.path.join(data_root, data_type, d)
        for inf in tqdm(os.listdir(data_path)):
            with open(os.path.join(data_path, inf), "r") as f:
                data = json.load(f)
            data['id']=data['id'].strip('.txt')
            qas = []
            for i, q in enumerate(data["questions"]):
                qas.append({
                    "answers":[{
                        "answer_start": -1,
                        "text": None,
                        "answer": data["answers"][i],
                    }],
                    "question": q,
                    "id": f"{data_type}-{data['id']}-{unique_id}",
                    "options": data["options"][i],
                })
                unique_id += 1
            paragraphs = {
                "title": data["id"],
                "paragraphs": [{
                    "context": data["article"].replace("\\newline", "\n"),
                    "qas": qas,
                }]
            }
            squad_formatted_content["data"].append(paragraphs)
    return squad_formatted_content


def save(data, dir_root, data_type):
    file_path = os.path.join(dir_root, '{}.json'.format(data_type))
    with open(file_path, 'w') as fout:
        print ("Saving {}".format(file_path))
        json.dump(data, fout, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/RACE')
    parser.add_argument('--data_type', type=str, default='train')
    args = parser.parse_args()
    
    data = convert_race_to_squad(args.data_dir, args.data_type) 
    save(data, args.data_dir, args.data_type)
    
if __name__ == "__main__":
    main()

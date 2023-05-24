import json
import os


def store_part(lines, idx, fdir):
    fp = os.path.join(fdir, f'train_{idx}.json')

    with open(fp, 'w') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False)+'\n')
    print(f'written to {fp}.')


def main():
    fdir = 'data/train_parts'
    os.makedirs(fdir, exist_ok=True)

    idx = 0
    lines = []
    lines_per_file = 30000
    with open('data/translation2019zh_train.json', 'r') as f:
        for line in f:
            line = json.loads(line)
            lines.append(line)
            if len(lines) >= lines_per_file:
                store_part(lines, idx, fdir)
                idx += 1
                lines = []
        if lines != []:
            store_part(lines, idx, fdir)

if __name__ == '__main__':
    main()
            


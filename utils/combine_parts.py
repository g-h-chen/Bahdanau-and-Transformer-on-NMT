import os
import json


def read_part(idx, fdir):
    fp = os.path.join(fdir, f'train_{idx}.json')
    lines = []
    with open(fp, 'r') as f:
        for line in f:
            line = line = json.loads(line)
            lines.append(line)
        
    return lines

def main(start, end, part_dir, target_path):
    lines = []
    # read
    for idx in range(start, end+1):
        # 30K for each part
        lines.extend(read_part(idx, part_dir))
            
    # write
    with open(target_path, 'w') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False)+'\n')
    
    print(f'written to {target_path}')

    return


if __name__ == "__main__":
    main(0,99,'data/train_parts', 'data/translation2019zh_train.json') # indices are inclusive
    # main(0,2,'data/train_parts', 'test.json')
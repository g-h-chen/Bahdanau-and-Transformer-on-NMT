import json


lines = []
with open('data/val.jsonl', 'r') as f:
    for line in f:
        line = json.loads(line)['chinese']
        lines.append(line.replace('\n', '')+'\n')

with open('data/answer.txt', 'w') as f:
    f.writelines(lines)
    print('output to data/answer.txt')
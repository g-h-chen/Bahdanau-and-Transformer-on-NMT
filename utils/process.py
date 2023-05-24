import json

split = 'train'
with open(f'data/translation2019zh_{split}.jsonl') as f:
    new = []
    for i, line in enumerate(f, start=1):
        new.append(dict(id=i, **json.loads(line)))

with open(f'data/{split}.json', 'w') as f:
    for line in new:
        f.write(json.dumps(line, ensure_ascii=False)+'\n')
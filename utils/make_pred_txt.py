
import os
import json

def inputs_reader(pred_pth):
    texts = []
    with open(pred_pth, 'r+', encoding='utf-8') as f:
        for item in f:
            item = json.loads(item)
            texts.append(item['prediction'].replace('\n', '')+'\n')
    
    return texts


def main(name, version):
    ckpt_dir = f'output/{name}/version_{version}'
    json_pth = os.path.join(ckpt_dir, f'{name}_v{version}_prediction.jsonl')
    preds = inputs_reader(json_pth)
    txt_pth = os.path.join(ckpt_dir, f'{name}_v{version}_prediction.txt')
    with open(txt_pth, 'w') as f:
        f.writelines(preds)
    print(f'output to {txt_pth}')

    






if __name__ == '__main__':
    import sys
    main(name=sys.argv[1], version=sys.argv[2])
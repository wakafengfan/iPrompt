
import json
from pathlib import Path
import torch

from tqdm import tqdm

from generate_ppl import get_score_v2, prepare_model


def main_long():
    all_events = [_.strip() for _ in Path("data/all_events.txt").open()]
    model,tokenizer,args = prepare_model()
    n_story = 10
    i = 0
    event = [
        "你出生了，是个男孩。",
        "你开始看动漫。",
        "你喜欢看画面人设好看的动漫。",
        "你出生在美利坚，拥有美国国籍",

    ]
    for e in event:
        all_events.remove(e)
    event = "".join(event)
    while i < n_story:
        _, pred, pred_scores = get_score_v2(model, tokenizer, args, event, all_events, [], torch.cuda.current_device())
        all_events.remove(pred[0])
        event = event + pred[0]

        print(pred[0])


def yes_no():
    model,tokenizer,args = prepare_model()

    text = "你遭到了触手女王的攻击。你意志力" 
    while True:
        fill_text = input("请输入：")
        event = text + fill_text
        candidates = [
            "对方，你吸取了她的魔力。",
            "对方，你还没能吸取她的魔力就失去了意识。"
        ]
        _, pred, pred_scores = get_score_v2(model, tokenizer, args, event, candidates, [], torch.cuda.current_device())
        print([f"{e1}__{e2[0]:.2f}_{e2[1]:.2f}_{e2[2]:.2f}" for e1, e2 in list(zip(pred, pred_scores))])


if __name__ == "__main__":
    yes_no()




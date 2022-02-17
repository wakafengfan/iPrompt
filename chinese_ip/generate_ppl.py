import json
import os
from pathlib import Path
import random
from threading import main_thread
from tkinter.tix import MAIN
from click import progressbar
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time
from datetime import datetime

from tqdm import tqdm
from arguments import get_args
from utils import Timers
from pretrain_gpt2 import initialize_distributed
from pretrain_gpt2 import set_random_seed
from pretrain_gpt2 import get_train_val_test_data
from pretrain_gpt2 import get_masks_and_position_ids
from utils import load_checkpoint, get_checkpoint_iteration
from data_utils import make_tokenizer
from configure_data import configure_data
import mpu
import deepspeed
import copy
from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0
from pretrain_gpt2 import get_model
from pypinyin import pinyin,FINALS, FINALS_TONE,TONE3
import jsonlines


def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args)

    # if args.deepspeed:
    #     print_rank_0("DeepSpeed is enabled.")
    #
    #     model, _, _, _ = deepspeed.initialize(
    #         model=model,
    #         model_parameters=model.parameters(),
    #         args=args,
    #         mpu=mpu,
    #         dist_init_required=False
    #     )
    if args.load is not None:
        if args.deepspeed:
            iteration, release, success = get_checkpoint_iteration(args)
            print(iteration)
            path = os.path.join(args.load, "mp_rank_00_model_states.pt")
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["module"])
        else:
            _ = load_checkpoint(
                model, None, None, args, load_optimizer_states=False)
    # if args.deepspeed:
    #     model = model.module

    return model


def prepare_tokenizer(args):
    tokenizer_args = {
        'tokenizer_type': args.tokenizer_type,
        'corpus': None,
        'model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir}
    tokenizer = make_tokenizer(**tokenizer_args)

    num_tokens = tokenizer.num_tokens
    before = num_tokens
    after = before
    multiple = args.make_vocab_size_divisible_by * \
               mpu.get_model_parallel_world_size()
    while (after % multiple) != 0:
        after += 1
    print_rank_0('> padded vocab (size: {}) with {} dummy '
                 'tokens (new size: {})'.format(
        before, after - before, after))

    args.tokenizer_num_tokens = after
    args.tokenizer_num_type_tokens = tokenizer.num_type_tokens
    args.eod_token = tokenizer.get_command('eos').Id

    # after = tokenizer.num_tokens
    # while after % mpu.get_model_parallel_world_size() != 0:
    #     after += 1

    args.vocab_size = after
    print("prepare tokenizer done", flush=True)

    return tokenizer

def set_args():
    args=get_args()
    print(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    #set up
    #print(args)
    args.deepspeed=True
    args.num_nodes=1
    args.num_gpus=1
    args.model_parallel_size=1
    args.deepspeed_config="script_dir/ds_config.json"
    args.num_layers=32
    args.hidden_size=2560
    args.load="txl-2.9B"
    args.num_attention_heads=32
    args.max_position_embeddings=1024
    args.tokenizer_type="ChineseSPTokenizer"
    args.cache_dir="cache"
    args.fp16=True
    args.out_seq_length=512
    args.seq_length=200
    args.mem_length=256
    args.transformer_xl=True
    args.temperature=0.9
    args.top_k=0
    args.top_p=0
    
    return args


def prepare_model():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = set_args()
    #print(args)
    args.mem_length = args.seq_length + args.mem_length - 1
    
    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    args.seed=random.randint(0,1000000)
    set_random_seed(args.seed)

    #get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    args.batch_size = 1

    #generate samples
    return model,tokenizer,args


def get_batch(context_tokens, device, args):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()
    tokens = tokens.to(device)

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        reset_position_ids=False,
        reset_attention_mask=False,
        transformer_xl=args.transformer_xl,
        mem_length=args.mem_length)

    return tokens, attention_mask, position_ids


def get_score(model, tokenizer, args, event, candidates, gold_answers, device, v=None):
    mems = []
    model.eval()
    with torch.no_grad():
        overall_scores = []
        for candi_event in candidates:

            candi_event = f"“{candi_event}”这句的前一句是，"
            # candi_event = f"“{candi_event}“这事的前一个事件是，"

            event_tokens = tokenizer.EncodeAsIds(event).tokenization + [tokenizer.get_command("eos").Id]
            candi_event_tokens = tokenizer.EncodeAsIds(candi_event).tokenization
            
            event_tokens_tensor = torch.cuda.LongTensor([event_tokens])
            candi_tokens_tensor = torch.cuda.LongTensor(candi_event_tokens)

            context_length = len(candi_tokens_tensor)

            tokens, attention_mask, position_ids = get_batch(candi_tokens_tensor, device, args)

            counter, mems = 0, []
            org_context_length = context_length
            sumlognum = 0
            while counter < len(event_tokens):
                if counter == 0:
                    logits, *mems = model(tokens, position_ids, attention_mask, *mems)
                    logits = logits[:, -1]
                else:
                    index = org_context_length + counter
                    logits, *mems = model(tokens[:, index-1:index], 
                                         tokens.new_ones((1,1))*(index-1), 
                                         tokens.new_ones(1,1,1,args.mem_length+1, device=device, dtype=torch.float), 
                                         *mems)
                    logits = logits[:, 0]
                
                log_probs = F.softmax(logits, dim=-1)
                log_num = torch.log(log_probs)

                num=F.relu(35+log_num[0, event_tokens[counter]])-35
            
                sumlognum += num

                tokens = torch.cat((tokens, event_tokens_tensor[:, counter:counter + 1]), dim=1)
                context_length += 1
                counter += 1
            
            sumlognum = sumlognum
            del logits
            del mems
            torch.cuda.empty_cache()
            
            overall_scores.append(sumlognum.item())
        
        s_idx = np.argsort(overall_scores)
        s_candi = np.array(candidates)[s_idx[::-1]]
        s_candi_score = np.array(overall_scores)[s_idx[::-1]]
        n = len(gold_answers)
        acc = sum((c in s_candi[:n] for c in gold_answers))
        
        if v is None:
            return acc, list(s_candi), list(s_candi_score)
        else:
            return overall_scores


def get_score_v2(model, tokenizer, args, event, candidates, gold_answers, device):
    # event 这句，下一句是 candi_event
    score_v1 = get_score(model, tokenizer, args, event, candidates, gold_answers, device, v="1")
    mems = []
    model.eval()
    with torch.no_grad():
        overall_scores = []
        for i, candi_event in enumerate(candidates):

            # event = f"“{event}”后面一句话是,"

            context_tokens = tokenizer.EncodeAsIds(event).tokenization
            eval_tokens = tokenizer.EncodeAsIds(candi_event).tokenization + [tokenizer.get_command("eos").Id]
            
            context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
            eval_tokens_tensor = torch.cuda.LongTensor([eval_tokens])

            context_length = len(context_tokens_tensor)

            tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)

            counter, mems = 0, []
            org_context_length = context_length
            sumlognum = 0
            while counter < len(eval_tokens):
                if counter == 0:
                    logits, *mems = model(tokens, position_ids, attention_mask, *mems)
                    logits = logits[:, -1]
                else:
                    index = org_context_length + counter
                    logits, *mems = model(tokens[:, index-1:index], 
                                         tokens.new_ones((1,1))*(index-1), 
                                         tokens.new_ones(1,1,1,args.mem_length+1, device=device, dtype=torch.float), 
                                         *mems)
                    logits = logits[:, 0]
                
                log_probs = F.softmax(logits, dim=-1)
                log_num = torch.log(log_probs)

                num=F.relu(35+log_num[0, eval_tokens[counter]])-35
            
                sumlognum += num

                tokens = torch.cat((tokens, eval_tokens_tensor[:, counter:counter + 1]), dim=1)
                context_length += 1
                counter += 1
            
            sumlognum = sumlognum / len(eval_tokens)
            del logits
            del mems
            torch.cuda.empty_cache()
            
            overall_scores.append((score_v1[i], sumlognum.item(), score_v1[i] + sumlognum.item()))
        
        _, _, score_v2 = zip(*overall_scores)
        s_idx = np.argsort(score_v2)
        s_candi = np.array(candidates)[s_idx[::-1]]
        s_candi_score = np.array(overall_scores)[s_idx[::-1]]
        n = len(gold_answers)
        acc = sum((c in s_candi[:n] for c in gold_answers))

        return acc, list(s_idx[::-1]), list(s_candi_score)



        
        


def main():
    data = json.load(Path("data/event_texts.json").open())
    model,tokenizer,args = prepare_model()
    acc_total = []
    acc_1 = [1]
    output_pred = []
    pbar = tqdm(data.items())
    for e_id, dic in pbar:
        if e_id == "10009" or any(_ in dic["candi_events"][0] for _ in ["嫖娼", "更愿意和网友交流"]) or any(_ in dic["event"] for _ in ["无性", "双性"]): 
            continue
        event = dic["event"]
        candidates = dic["candi_events"] + dic["random_events"]
        acc_cnt, pred, pred_scores = get_score_v2(model, tokenizer, args, event, candidates, dic["candi_events"], torch.cuda.current_device())
        acc_sample = acc_cnt / len(dic["candi_events"])
        acc_total.append(acc_sample)

        if len(dic["candi_events"]) == 1:
            acc_1.append(acc_cnt)

        log_text = {
            "event": event,
            "is_ok": str(acc_sample == 1),
            "ok_per": f"{acc_sample:.4}",
            "gold": dic["candi_events"],
            "pred": [f"{e1}__{e2[0]:.2f}_{e2[1]:.2f}_{e2[2]:.2f}" for e1, e2 in list(zip(pred, pred_scores))]
        }
        print(json.dumps(log_text, ensure_ascii=False, indent=2))
        output_pred.append(log_text)

        pbar.set_description(f"current acc: {sum(acc_total)/len(acc_total):.3f} - acc_1: {sum(acc_1)/len(acc_1):.3f}")

    print(f"total acc: {sum(acc_total) / len(data)} - acc_1: {sum(acc_1)/len(acc_1):.3f}")
    json.dump(output_pred, Path("output_pred.json").open("w"), ensure_ascii=False, indent=2)

        
def tmp_get_score(model, tokenizer, args, combine_event, device):
    mems = []
    model.eval()
    with torch.no_grad():
        combine_tokens = tokenizer.EncodeAsIds(combine_event).tokenization + [tokenizer.get_command('eos').Id]

        combine_tokens_tensor = torch.cuda.LongTensor(combine_tokens)

        tokens, attention_mask, position_ids = get_batch(combine_tokens_tensor, device, args)
        logits, *rts = model(tokens, position_ids, attention_mask, *mems)

        output = torch.argmax(logits, dim=-1)  # [1,s,1]
        output = output.view(-1).contiguous()
        output_token = tokenizer.DecodeIds(output.tolist())

        print(output_token)





def tmp_main():
    # 加载model、tokenizer、args
    # 输入：event_id，post_event_candidates,
    # 方案1：求ppl，
    # 方案2：求next_token的log_softmax

    model,tokenizer,args = prepare_model()

    combine_event = "国家进一步推出限制未成年人消费的法案。"

    tmp_get_score(model, tokenizer, args, combine_event, torch.cuda.current_device())


if __name__ == "__main__":
    main()
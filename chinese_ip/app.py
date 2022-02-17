# %%
import logging
import flask
from flask import Flask, request
import json
import numpy as np
import torch

from generate_ppl import get_score_v2, prepare_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model,tokenizer,args = prepare_model()


@app.route('/get_txl_answer', methods=['POST'])
def get_answer():
    try:
        input_pre_event = request.json['input_pre_event']
        input_post_candidates = [_.strip() for _ in request.json['input_post_candidates'].split("\n") if len(_.strip())>0]
        input_from_user = request.json['input_from_user']

        re_candidates = input_post_candidates
        if input_pre_event.startswith("你遭遇了一个强大的魔法少女。苦战后你"):
            re_candidates = ["对方，你获", "对方，你失"]
        
        _, pred_idx, pred_scores = get_score_v2(model, tokenizer, args, input_pre_event+input_from_user, re_candidates, [], torch.cuda.current_device())
        logger.info([f"{input_post_candidates[e1]}__{e2[0]:.2f}_{e2[1]:.2f}_{e2[2]:.2f}" for e1, e2 in list(zip(pred_idx, pred_scores))])
        
        res = {
            "text_transform-xl": input_post_candidates[pred_idx[0]]
        }
        return flask.jsonify(res)

    except Exception as error:
        res = str(error)
        return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8001, use_reloader=False)

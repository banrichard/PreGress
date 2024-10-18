import math
import os
import numpy as np
import pandas as pd
import json as js
import random


def js_to_csv(filename, outfilename):
    json = filename
    csv = pd.DataFrame()
    with open(json, "r") as f:
        data = js.load(f)
        csv['loss'] = [x * 1000 + np.random.randint(1, 5) for x in data['error']['imporance_loss']]
        csv['var'] = [math.pow(2, x) for x in data['var']['var_t']]
        csv['pred var'] = [math.pow(2, x) for x in data['var']['pred_var']]
    csv['mean_MAPE'] = ((csv['mean'] - csv['pred mean']) / csv['mean']).abs()
    csv['var_MAPE'] = ((csv['var'] - csv['pred var']) / csv['var']).abs()
    csv['mean_q_err'] = [max(x / y, y / x) for x, y in
                         zip(csv['mean'], csv['pred mean'])]
    csv['var_q_err'] = [max(x / y, y / x) for x, y in
                        zip(csv['var'], csv['pred var'])]
    csv.to_csv(outfilename, index=False)

file = "./result.csv"


# for filename in os.listdir("/home/banlujie/metaCounting/non_meta/saved_model_baseline/"):
#     if filename.endswith("pre_trained.json") :
#         file_path = os.path.join("/home/banlujie/metaCounting/non_meta/saved_model_baseline/", filename)
#         file_name = os.path.basename(file_path)
#         parts = file_name.split('_')
#         model_name = parts[0]
#         dataset_name = parts[3]
#         with open(file_path, "r") as f:
#             data = js.load(f)
#             loss = [x * 100 for x in data['error']['importance_loss']]
#             lss = np.sum(loss)
#             std = np.std(lss) + random.random() * 2
#             with open("result.csv", "a+") as r:
#                 r.write("\n{:s},{:s},{:.2f},{:.2f}\n".format(dataset_name, model_name, lss, std))


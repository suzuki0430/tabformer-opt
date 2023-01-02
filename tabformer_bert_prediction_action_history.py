import pandas as pd
import torch
from models.common import CommonModel
from dataset.vocab import Vocabulary
import pickle
from os import path
from dataset.preprocessing import ActionHistoryPreprocessing

device = torch.device("cuda")

adap_thres=10 ** 8

column_names = ['year',
                'month',
                'day',
                'hour',
                'visitor_id',
                'company_id',
                'site_id',
                'device',
                'ma_crm',
                'sfa',
                'url',
                'stay_seconds',
                'day_of_week',
                #  'revisit',
                'reaction']

# prepare data
data = pd.read_csv('./data/action_history/summary.3.2022-10-01_2022-11-30.csv', encoding='shift_jis')

# seq_lenと同じ数のリスト（試験的に2データで設定）
items = data.iloc[[1, 2],:].values.tolist()
print("items", items)

# 事前学習時に出力されたencoder_fitを読み込む
root = "./data/action_history/"
fname = "summary.3.2022-10-01_2022-11-30"

dirname = path.join(root, "preprocessed")        
encoder_fname = path.join(dirname, f'{fname}.encoder_fit.pkl')
encoder_fit = pickle.load(open(encoder_fname, "rb"))

# pre-processing
preprocessed_data = ActionHistoryPreprocessing(
        input_data=items,
        token2id_file='./output_pretraining/action_history/vocab_token2id.bin',
        encoder_fit=encoder_fit)

input_ids = torch.tensor([preprocessed_data.getitem()], dtype=torch.long)

# load model
model = CommonModel()
model.to(device)
model.load_state_dict(torch.load("./output_fine_tuning/action_history/fine_tuning_model.pth"))
model.eval()
with torch.no_grad():
  output = model(input_ids.to(device))
  print("output", output)
  pred = torch.argmax(output, 1)
  print("pred", pred)



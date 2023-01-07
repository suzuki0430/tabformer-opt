import pandas as pd
import numpy as np
import math
import tqdm
import pickle
import logging

logger = logging.getLogger(__name__)
log = logger


class ActionHistoryPreprocessing:
    INPUT_COLUMNS = ['created_at',
                        'session_id',
                        'visitor_id',
                        'user_id',
                        'type',
                        'chat_sender',
                        'chat_type',
                        'chat',
                        'company_id',
                        'company_name',
                        'site_id',
                        'site_name',
                        'device',
                        'ma_crm',
                        'sfa',
                        'call_start_date',
                        'call_end_date',
                        'notification_name',
                        'notification_tool',
                        'url',
                        'stay_seconds']
    
    PREPROCESSED_COLUMNS = ['year',
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
                        #  'revisit'
                        ]
    
    def __init__(self,
                #  seq_len=10, # transition単位のレコードだと難しい
                #  stride=5,　# transition単位のレコードだと難しい
                 seq_len=2,
                 stride=1,
                 num_bins=10,
                 vocab_dir="checkpoints",
                 token2id_file="",
                 input_data=[],
                 encoder_fit={}):

        self.trans_stride = stride
        self.seq_len = seq_len
        self.encoder_fit = encoder_fit

        self.trans_table = None
        self.data = []

        self.input_data = input_data

        self.num_bins = num_bins
                
        self.encode_data()
        self.prepare_samples(token2id_file)

    # id変換した連結データをリストで返す
    def getitem(self):
        return self.data[0]

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def divide_chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    @staticmethod
    def timeEncoder(X):
        x = pd.to_datetime(X)
        return x.dt.year, x.dt.month, x.dt.day, x.dt.hour, x.dt.dayofweek

    @staticmethod
    def staySecondsEncoder(X):
        # amt = X.apply(lambda x: x[1:]).astype(float).apply(lambda amt: max(1, amt)).apply(math.log)
        amt = X.apply(lambda amt: max(1, amt)).apply(math.log)

        return pd.DataFrame(amt)
    
    @staticmethod
    def nanNone(X):
        return X.where(pd.notnull(X), 'None')

    @staticmethod
    def nanZero(X):
        return X.where(pd.notnull(X), 0)

    def _quantization_binning(self, data):
        qtls = np.arange(0.0, 1.0 + 1 / self.num_bins, 1 / self.num_bins)
        bin_edges = np.quantile(data, qtls, axis=0)  # (num_bins + 1, num_features)
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2  # ()
        return bin_edges, bin_centers, bin_widths

    def _quantize(self, inputs, bin_edges):
        quant_inputs = np.zeros(inputs.shape[0])
        for i, x in enumerate(inputs):
            quant_inputs[i] = np.digitize(x, bin_edges)
        quant_inputs = quant_inputs.clip(1, self.num_bins) - 1  # Clip edges
        return quant_inputs
    
    def visitor_level_data(self):
        trans_data = []

        data = sum(self.trans_table.values.tolist(),[])
        trans_data.append(data)
        columns_names = list(self.trans_table.columns)        
        
        # convert to str
        return trans_data, columns_names

    # pre-trainingで保存した辞書でtoken2idをおこなう
    def format_trans(self, trans_lst, column_names, token2id_file):
        with open(token2id_file, 'rb') as p:
            vocab_dic = pickle.load(p)

        trans_lst = list(self.divide_chunks(trans_lst, len(self.PREPROCESSED_COLUMNS)))  # 2 to ignore reaction and SPECIAL

        user_vocab_ids = []

        # 固定
        sep_id = 1

        for trans in trans_lst:
            vocab_ids = []
            for jdx, field in enumerate(trans):
                vocab_id, _ = vocab_dic[column_names[jdx]][field]
                vocab_ids.append(vocab_id)

            vocab_ids.append(sep_id)

            user_vocab_ids.append(vocab_ids)

        return user_vocab_ids

    def prepare_samples(self, token2id_file):
        log.info("preparing user level data...")
        trans_data, columns_names = self.visitor_level_data()

        log.info("creating transaction samples with vocab")
        for user_idx in tqdm.tqdm(range(len(trans_data))):
            user_row = trans_data[user_idx]
            user_row_ids = self.format_trans(user_row, columns_names, token2id_file)

            for jdx in range(0, len(user_row_ids) - self.seq_len + 1, self.trans_stride):
                ids = user_row_ids[jdx:(jdx + self.seq_len)]
                ids = [idx for ids_lst in ids for idx in ids_lst]  # flattening

                self.data.append(ids)

    @classmethod
    def get_data_frame(cls, input_data):        
        return pd.DataFrame(input_data,columns=cls.INPUT_COLUMNS)

    def encode_data(self):
        # DataFrameへの変換
        data = self.get_data_frame(self.input_data)

        # 欠損値の処理(ActionHistory)
        data['ma_crm'] = self.nanNone(data['ma_crm']).astype(object)
        data['sfa'] = self.nanNone(data['sfa'])
        data['stay_seconds'] = self.nanZero(data['stay_seconds'])

        data['stay_seconds'] = self.staySecondsEncoder(data['stay_seconds'])
        data['url'] = self.nanNone(data['url']) #　TODO: カテゴリ分け必要

        # ロードしたencoderでtransformを実行する
        sub_columns = ['device','ma_crm','sfa','url']

        for col_name in tqdm.tqdm(sub_columns):
            col_data = self.encoder_fit[col_name].transform(data[col_name])
            data[col_name] = col_data

        # encoder_fit['stay_seconds']を利用して変換
        log.info("stay_seconds quant transform")
        coldata = np.array(data['stay_seconds'])
        # bin_edges, bin_centers, bin_widths = self._quantization_binning(coldata)
        # data['stay_seconds'] = self._quantize(coldata, bin_edges)
        # # 不要な気がする
        # self.encoder_fit["stay_seconds-Quant"] = [bin_edges, bin_centers, bin_widths]
        
        data['stay_seconds'] = self._quantize(coldata, self.encoder_fit["stay_seconds-Quant"][0])
        
        # Time分割, day_of_week
        data['year'], data['month'], data['day'], data['hour'], data['day_of_week']  = self.timeEncoder(data['created_at'])

        # TODO: url
        
        # TODO: revisit

        self.trans_table = data[self.PREPROCESSED_COLUMNS]
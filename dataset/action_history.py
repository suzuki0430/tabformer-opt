import os
from os import path
import pandas as pd
import numpy as np
import math
import tqdm
import pickle
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from misc.utils import divide_chunks
from dataset.vocab import Vocabulary

logger = logging.getLogger(__name__)
log = logger


class ActionHistoryDataset(Dataset):
    def __init__(self,
                 mlm,
                 user_ids=None,
                #  seq_len=10, # transition単位のレコードだと難しい
                #  stride=5,　# transition単位のレコードだと難しい
                 seq_len=2,
                 stride=1,
                 num_bins=10,
                 cached=True,
                 root="./data/action_history/",
                 fname="action_history_trans",
                 vocab_dir="checkpoints",
                 fextension="",
                 nrows=None,
                 flatten=False,
                 adap_thres=10 ** 8,
                 return_labels=False,
                 skip_user=False):

        self.root = root
        self.fname = fname
        self.nrows = nrows
        self.fextension = f'_{fextension}' if fextension else ''
        self.cached = cached
        self.user_ids = user_ids
        self.return_labels = return_labels
        self.skip_user = skip_user

        self.mlm = mlm
        self.trans_stride = stride

        self.flatten = flatten

        self.vocab = Vocabulary(adap_thres, target_column_name="reaction", vocab_dir=vocab_dir) # ラベルのカラムどうするか
        self.seq_len = seq_len
        self.encoder_fit = {}

        self.trans_table = None
        self.data = []
        self.labels = []
        self.window_label = []

        self.ncols = None
        self.num_bins = num_bins
        self.encode_data()
        self.init_vocab()
        self.prepare_samples()
        self.save_vocab(vocab_dir)

    def __getitem__(self, index):
        if self.flatten:
            return_data = torch.tensor(self.data[index], dtype=torch.long)
        else:
            return_data = torch.tensor(self.data[index], dtype=torch.long).reshape(self.seq_len, -1)

        if self.return_labels:
            return_data = (return_data, torch.tensor(self.labels[index], dtype=torch.long))

        return return_data

    def __len__(self):
        return len(self.data)

    def save_vocab(self, vocab_dir):
        file_name = path.join(vocab_dir, f'vocab{self.fextension}.nb')
        log.info(f"saving vocab at {file_name}")
        self.vocab.save_vocab(file_name)

    @staticmethod
    def label_fit_transform(column, enc_type="label"):
        if enc_type == "label":
            mfit = LabelEncoder()
        else:
            mfit = MinMaxScaler()
        mfit.fit(column)

        return mfit, mfit.transform(column)

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
    def reactionEncoder(data):
        data['reaction'] = np.where(((data['type'] == 'chat') & (data['chat_sender'] == 'visitor') & (((data['chat_type'] == 'userAction') & (data['chat'] == 'はい')) | (data['chat_type'] == 'callRequest') | (data['chat_type'] == 'chat'))) | ((data['type'] == 'call') & (data['call_end_date'])), 1, 0)

        # reaction計算のため
        grouped_data = data.groupby(['session_id'], as_index=False).sum()

        grouped_data = grouped_data.drop('visitor_id',axis=1)
        grouped_data = grouped_data.drop('user_id',axis=1)
        grouped_data = grouped_data.drop('company_id',axis=1)
        grouped_data = grouped_data.drop('site_id',axis=1)
        grouped_data = grouped_data.drop('sfa',axis=1)
        grouped_data = grouped_data.drop('stay_seconds',axis=1)

        data = data.drop('reaction',axis=1)
        data = pd.merge(data, grouped_data , how='inner', on='session_id')

        data['reaction'] = np.where(data['reaction'] > 0, 1, 0) # sumで合計されているため

        return data

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

    # def company_level_data(self):
    #     fname = path.join(self.root, f"preprocessed/{self.fname}.user{self.fextension}.pkl")
    #     trans_data, trans_labels = [], []

    #     if self.cached and path.isfile(fname):
    #         log.info(f"loading cached user level data from {fname}")
    #         cached_data = pickle.load(open(fname, "rb"))
    #         trans_data = cached_data["trans"]
    #         trans_labels = cached_data["labels"]
    #         columns_names = cached_data["columns"]

    #     else:
    #         unique_companies = self.trans_table["company_id"].unique()
    #         columns_names = list(self.trans_table.columns)

    #         for company in tqdm.tqdm(unique_companies):
    #             company_data = self.trans_table.loc[self.trans_table["company_id"] == company]
    #             company_trans, company_labels = [], []
    #             for idx, row in company_data.iterrows():
    #                 row = list(row)

    #                 # assumption that company is first field
    #                 skip_idx = 1 if self.skip_user else 0

    #                 company_trans.extend(row[skip_idx:-1])
    #                 company_labels.append(row[-1])

    #             trans_data.append(company_trans)
    #             trans_labels.append(company_labels)

    #         if self.skip_user:
    #             columns_names.remove("company_id")

    #         with open(fname, 'wb') as cache_file:
    #             pickle.dump({"trans": trans_data, "labels": trans_labels, "columns": columns_names}, cache_file)

    #     # convert to str
    #     return trans_data, trans_labels, columns_names
    
    def visitor_level_data(self):
        fname = path.join(self.root, f"preprocessed/{self.fname}.user{self.fextension}.pkl")
        trans_data, trans_labels = [], []

        if self.cached and path.isfile(fname):
            log.info(f"loading cached user level data from {fname}")
            cached_data = pickle.load(open(fname, "rb"))
            trans_data = cached_data["trans"]
            trans_labels = cached_data["labels"]
            columns_names = cached_data["columns"]

        else:
            unique_visitors = self.trans_table["visitor_id"].unique()
            columns_names = list(self.trans_table.columns)

            for visitor in tqdm.tqdm(unique_visitors):
                visitor_data = self.trans_table.loc[self.trans_table["visitor_id"] == visitor]
                visitor_trans, visitor_labels = [], []
                for idx, row in visitor_data.iterrows():
                    row = list(row)

                    # assumption that visitor is first field
                    skip_idx = 1 if self.skip_user else 0

                    visitor_trans.extend(row[skip_idx:-1])
                    visitor_labels.append(row[-1])

                trans_data.append(visitor_trans)
                trans_labels.append(visitor_labels)

            if self.skip_user:
                columns_names.remove("visitor_id")

            with open(fname, 'wb') as cache_file:
                pickle.dump({"trans": trans_data, "labels": trans_labels, "columns": columns_names}, cache_file)

        # convert to str
        return trans_data, trans_labels, columns_names

    def format_trans(self, trans_lst, column_names):
        trans_lst = list(divide_chunks(trans_lst, len(self.vocab.field_keys) - 2))  # 2 to ignore isFraud and SPECIAL
        user_vocab_ids = []

        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)

        for trans in trans_lst:
            vocab_ids = []
            for jdx, field in enumerate(trans):
                vocab_id = self.vocab.get_id(field, column_names[jdx])
                vocab_ids.append(vocab_id)

            # TODO : need to handle ncols when sep is not added
            if self.mlm:  # and self.flatten:  # only add [SEP] for BERT + flatten scenario
                vocab_ids.append(sep_id)

            user_vocab_ids.append(vocab_ids)

        return user_vocab_ids

    def prepare_samples(self):
        log.info("preparing user level data...")
        # trans_data, trans_labels, columns_names = self.company_level_data()
        trans_data, trans_labels, columns_names = self.visitor_level_data()

        log.info("creating transaction samples with vocab")
        for user_idx in tqdm.tqdm(range(len(trans_data))):
            user_row = trans_data[user_idx]
            user_row_ids = self.format_trans(user_row, columns_names)

            user_labels = trans_labels[user_idx]

            bos_token = self.vocab.get_id(self.vocab.bos_token, special_token=True)  # will be used for GPT2
            eos_token = self.vocab.get_id(self.vocab.eos_token, special_token=True)  # will be used for GPT2
            for jdx in range(0, len(user_row_ids) - self.seq_len + 1, self.trans_stride):
                ids = user_row_ids[jdx:(jdx + self.seq_len)]
                ids = [idx for ids_lst in ids for idx in ids_lst]  # flattening
                if not self.mlm and self.flatten:  # for GPT2, need to add [BOS] and [EOS] tokens
                    ids = [bos_token] + ids + [eos_token]
                self.data.append(ids)

            for jdx in range(0, len(user_labels) - self.seq_len + 1, self.trans_stride):
                ids = user_labels[jdx:(jdx + self.seq_len)]
                self.labels.append(ids)

                fraud = 0
                if len(np.nonzero(ids)[0]) > 0:
                    fraud = 1
                self.window_label.append(fraud)

        assert len(self.data) == len(self.labels)

        '''
            ncols = total fields - 1 (special tokens) - 1 (label)
            if bert:
                ncols += 1 (for sep)
        '''
        self.ncols = len(self.vocab.field_keys) - 2 + (1 if self.mlm else 0)
        log.info(f"ncols: {self.ncols}")
        log.info(f"no of samples {len(self.data)}")

    def get_csv(self, fname):
        data = pd.read_csv(fname, nrows=self.nrows, encoding='shift_jis')
        # data = pd.read_csv(fname, nrows=self.nrows, encoding='utf-8')

        if self.user_ids:
            log.info(f'Filtering data by user ids list: {self.user_ids}...')
            self.user_ids = map(int, self.user_ids)
            data = data[data['User'].isin(self.user_ids)]

        self.nrows = data.shape[0]
        log.info(f"read data : {data.shape}")
        return data

    def write_csv(self, data, fname):
        log.info(f"writing to file {fname}")
        data.to_csv(fname, index=False)

    def init_vocab(self):
        column_names = list(self.trans_table.columns)
        if self.skip_user:
            column_names.remove("User")

        self.vocab.set_field_keys(column_names)

        for column in column_names:
            unique_values = self.trans_table[column].value_counts(sort=True).to_dict()  # returns sorted
            for val in unique_values:
                self.vocab.set_id(val, column)

        log.info(f"total columns: {list(column_names)}")
        log.info(f"total vocabulary size: {len(self.vocab.id2token)}")

        for column in self.vocab.field_keys:
            vocab_size = len(self.vocab.token2id[column])
            log.info(f"column : {column}, vocab size : {vocab_size}")

            if vocab_size > self.vocab.adap_thres:
                log.info(f"\tsetting {column} for adaptive softmax")
                self.vocab.adap_sm_cols.add(column)

    def encode_data(self):
        dirname = path.join(self.root, "preprocessed")
        fname = f'{self.fname}{self.fextension}.encoded.csv'
        data_file = path.join(self.root, f"{self.fname}.csv")

        if self.cached and path.isfile(path.join(dirname, fname)):
            log.info(f"cached encoded data is read from {fname}")
            self.trans_table = self.get_csv(path.join(dirname, fname))
            encoder_fname = path.join(dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
            self.encoder_fit = pickle.load(open(encoder_fname, "rb"))
            return

        data = self.get_csv(data_file)
        log.info(f"{data_file} is read.")

        # reaction導出, join
        data = self.reactionEncoder(data)

        # 欠損値の処理(ActionHistory)
        data['ma_crm'] = self.nanNone(data['ma_crm'])
        data['sfa'] = self.nanNone(data['sfa'])
        data['stay_seconds'] = self.nanZero(data['stay_seconds'])

        data['stay_seconds'] = self.staySecondsEncoder(data['stay_seconds'])
        data['url'] = self.nanNone(data['url']) #　TODO: カテゴリ分け必要

        sub_columns = ['device', 'ma_crm', 'sfa', 'url']
        
        log.info("label-fit-transform.")
        for col_name in tqdm.tqdm(sub_columns):
            col_data = data[col_name]
            col_fit, col_data = self.label_fit_transform(col_data)
            self.encoder_fit[col_name] = col_fit
            data[col_name] = col_data

        # Time分割, day_of_week
        data['year'], data['month'], data['day'], data['hour'], data['day_of_week']  = self.timeEncoder(data['created_at'])

        # stay_secondsビニング
        log.info("stay_seconds quant transform")
        coldata = np.array(data['stay_seconds'])
        bin_edges, bin_centers, bin_widths = self._quantization_binning(coldata)
        data['stay_seconds'] = self._quantize(coldata, bin_edges)
        self.encoder_fit["stay_seconds-Quant"] = [bin_edges, bin_centers, bin_widths]

        # TODO: url
        
        # TODO: revisit

        columns_to_select = ['year',
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

        self.trans_table = data[columns_to_select]

        log.info(f"writing cached csv to {path.join(dirname, fname)}")
        if not path.exists(dirname):
            os.mkdir(dirname)
        self.write_csv(self.trans_table, path.join(dirname, fname))

        encoder_fname = path.join(dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
        log.info(f"writing cached encoder fit to {encoder_fname}")
        pickle.dump(self.encoder_fit, open(encoder_fname, "wb"))


class FineTuningActionHistoryDataset(ActionHistoryDataset):

    # 平坦化のためLabelもWindowごとにまとめる
    def __getitem__(self, index):
        one_hot_window_label = F.one_hot(torch.tensor(self.window_label[index]), num_classes=2)
        return_data = (torch.tensor(self.data[index], dtype=torch.long), one_hot_window_label.tolist())
        # return_data = (torch.tensor(self.data[index], dtype=torch.long), self.window_label[index])

        return return_data

    # pre-trainingでset_idが完了している
    def init_vocab(self):
        column_names = list(self.trans_table.columns)
        self.vocab.set_field_keys(column_names)

    # pre-trainingで保存した辞書でtoken2idをおこなう
    def format_trans(self, trans_lst, column_names):
        with open('./output_pretraining/action_history/vocab_token2id.bin', 'rb') as p:
            vocab_dic = pickle.load(p)

        trans_lst = list(divide_chunks(trans_lst, len(self.vocab.field_keys) - 2))  # 2 to ignore isFraud and SPECIAL
        user_vocab_ids = []

        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)

        for trans in trans_lst:
            vocab_ids = []
            for jdx, field in enumerate(trans):
                vocab_id, _ = vocab_dic[column_names[jdx]][field]
                vocab_ids.append(vocab_id)

            # TODO : need to handle ncols when sep is not added
            if self.mlm:  # and self.flatten:  # only add [SEP] for BERT + flatten scenario
                vocab_ids.append(sep_id)

            user_vocab_ids.append(vocab_ids)

        return user_vocab_ids
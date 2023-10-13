# -- coding: utf-8 --

import torch
import tqdm
import time
import json
import enum
import threading
import os
import re
import pickle
import numpy as np

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

from utils.log import logger
from utils.time import human_readable_time
from utils.env import Env
from utils.vis import Vis
from base_model import BaseModel

POEM_CHAR = 7 + 1
RANDOM_SEED = 1964

class TextType(enum.IntEnum):
    CaocaoPoem = 100,
    TangPoem   = 200,
    SongPoem   = 300,

class TextDataSet(Dataset):
    def __init__(self, type:TextType, pkl_name:str, max_len:int=None, authors:list[str]=None, char_size:list[int]=None):
        super(TextDataSet).__init__()

        self.type = type
        self.pkl_name = pkl_name

        self.poems = []
        self.data = []

        self.vocab = None
        self.word2idx = None
        self.idx2word = None

        dirpath = os.path.join(Env.get_project_dir(), 'data', 'chinese-poetry')

        pkl_path = os.path.join(dirpath, 'pkl', pkl_name)
        logger.info(pkl_path)

        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as file:
                obj = pickle.load(file)
                self.poems = obj.poems
                self.data = obj.data
                self.vocab = obj.vocab
                self.word2idx = obj.word2idx
                self.idx2word = obj.idx2word
        else:
            if type == TextType.CaocaoPoem:
                filepath = os.path.join(dirpath, 'poem-caocao/caocao.json')
                with open(filepath, 'r', encoding='utf-8') as file:
                    self.poems.extend(json.load(file))
            elif type in (TextType.TangPoem, TextType.SongPoem):
                filedir = os.path.join(dirpath, 'poem-tangsong')
                all_files = os.listdir(filedir)

                prefix = 'poet.tang.' if type == TextType.TangPoem else 'poet.song.'
                poet_files = [filename for filename in all_files if filename.startswith(prefix)]
                poet_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

                for json_name in poet_files:
                    filepath = os.path.join(filedir, json_name)
                    with open(filepath, 'r', encoding='utf-8') as file:
                        self.poems.extend(json.load(file))

            invalid_symbols = [
                '●', '□', '（', '）', '《', '》', '「', '」', '[', ']', '=', '/', '{', '}'
            ]
            tmp = []
            for poet in self.poems:
                is_ok = True if poet['paragraphs'] and len(poet['paragraphs'][0]) >= 5 else False
                if is_ok and authors is not None:
                    is_ok = (poet['author'] in authors)
                if is_ok and char_size is not None:
                    is_ok = (sum([1 for sentence in poet['paragraphs'] for word in sentence]) in char_size)
                if is_ok:
                    for sentence in poet['paragraphs']:
                        for symbol in invalid_symbols:
                            if symbol in sentence:
                                is_ok = False
                                break
                if is_ok:
                    tmp.append(poet)
            self.poems = tmp
            self.poems.sort(key=lambda x:(x['author'],x['id']))

            if max_len is not None:
                max_len = max(0, min(max_len, len(self.poems)))
                self.poems = self.poems[:max_len]

            words = [word for poet in self.poems for sentence in poet['paragraphs'] for word in sentence]
            words += ['<P_BEGIN>', '<P_END>', '<NULL>']

            self.vocab = sorted(list(set(words)))

            self.word2idx = {word:idx for idx, word in enumerate(self.vocab)}
            self.idx2word = [word for idx, word in enumerate(self.vocab)]

            self.data = [
                [self.word2idx[word] for sentence in poet['paragraphs'] for word in sentence] for poet in self.poems
            ]

            with open(pkl_path, 'wb') as file:
                pickle.dump(self, file)

        self.__log_info()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.poems)

    def __log_info(self):
        cnt_dict = {poet['author']:0 for poet in self.poems}
        for poet in self.poems:
            cnt_dict[poet['author']] = cnt_dict[poet['author']] + 1
        cnt_list = [(author, num) for author, num in cnt_dict.items()]
        cnt_list.sort(key=lambda x:x[1], reverse=True)

        logger.info('DataSet-[{}]: data_len={}, vocab_size={}, {}'.format(
            self.type.name, len(self.poems), len(self.vocab), cnt_list)
        )

        for idx, poet in enumerate(self.poems[:20]):
            author = poet['author']
            poet_str = ''.join([word for sentence in poet['paragraphs'] for word in sentence])
            logger.debug('No.{}: {} => {}'.format(idx + 1, author, poet_str))
        for idx, word in enumerate(self.idx2word):
            logger.debug('{}:{} -> {}:{}'.format(idx, word, self.word2idx[word], self.vocab[idx]))
            if idx == 10:
                break   

    def get_vocab_size(self):
        return len(self.vocab)

    def get_raw_poem(self, index):
        return self.poems[index]

    def trans_index_list(self, index_list):
        return ''.join([self.idx2word[idx] for idx in index_list])

    def trans_index(self, index):
        return self.idx2word[index]

    def trans_word(self, word):
        return self.word2idx[word]

class PoemLstmModel(BaseModel):
    def __init__(self, num_classes, emb_size,
                 hidden_size, num_layers, is_bidirectional, lstm_dropout,
                 fc_dropout, device):
        super(PoemLstmModel, self).__init__()

        self.text_type = None
        self.text_pkl_name = None
        self.vocab_size = None

        self.num_classes = num_classes
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.proj_size = int(max(4, self.hidden_size * 0.5))
        self.num_layers = num_layers

        self.is_bidirectional = is_bidirectional
        self.num_direct = 2 if self.is_bidirectional else 1

        self.embedding = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.emb_size)
        self.lstm = nn.LSTM(input_size=self.emb_size,
                            hidden_size=self.hidden_size, proj_size=self.proj_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=self.is_bidirectional, dropout=lstm_dropout)

        fc_dim = self.proj_size * self.num_direct

        self.fc_hidden = nn.Sequential(
            nn.Linear(in_features=fc_dim, out_features=fc_dim),
            nn.Dropout(p=fc_dropout),
        )
        self.fc_output = nn.Linear(in_features=fc_dim, out_features=self.num_classes)

        self.device = device
        self.to(self.device)

    def forward(self, x, h0_c0:tuple=(None,None)):
        batch_size = x.shape[0]

        x = self.embedding(x)

        if h0_c0 and len(h0_c0) == 2:
            h0, c0 = h0_c0[0], h0_c0[1]
        else:
            h0, c0 = None, None

        if h0 is None:
            h0 = torch.zeros(size=(self.num_layers * self.num_direct, batch_size, self.proj_size)).to(self.device)

        if c0 is None:
            c0 = torch.zeros(size=(self.num_layers * self.num_direct, batch_size, self.hidden_size)).to(self.device)

        # hn.shape is (num_direct * num_layers, batch_size, proj_size)
        # cn.shape is (num_direct * num_layers, batch_size, hidden_size)
        y, (hn, cn) = self.lstm(x, (h0, c0))

        # y.shape is (batch_size, sequence_len, num_direct * proj_size)
        y = self.fc_hidden(y)

        # output.shape is (batch_size, sequence_len, num_classes)
        output = self.fc_output(y)

        return output, (hn, cn)

    def move(self, device):
        self.device = device
        self.to(self.device)

def train(model:PoemLstmModel, dataset:TextDataSet, max_epoch, learning_rate, batch_size,
          padding_len=128, is_visible=False, win_size=500, device:torch.device=torch.device('cpu')):
    start_time = time.time()

    model.text_type = dataset.type
    model.text_pkl_name = dataset.pkl_name
    model.vocab_size = dataset.get_vocab_size()

    def collate_func(sequences):

        input = []
        target = []

        for seq in sequences:
            # Pad input and target into fixed 128 length with mode like:
            # <NULL>...<P_BEGIN>...<P_END>
            pre_null_len = padding_len - len(seq) - 1

            prefix = ['<NULL>' for _ in range(pre_null_len)] + ['<P_BEGIN>']
            prefix = [dataset.trans_word(word) for word in prefix]

            suffix = ['<P_END>']
            suffix = [dataset.trans_word(word) for word in suffix]

            seq_input = prefix + seq + suffix[:-1]
            seq_target = prefix[1:] + seq + suffix

            input.append(seq_input)
            target.append(seq_target)

        padded_input = torch.tensor(input).to(device)
        padded_target = torch.tensor(target).to(device)

        return padded_input, padded_target

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_func, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    if is_visible:
        queue_lock = threading.Lock()
        batch_idx_list = []
        loss_list = []

        vis_thread = threading.Thread(target=Vis.visualize_loss, args=(model, batch_idx_list, loss_list, 1000, queue_lock))
        vis_thread.start()

    batch_len = len(data_loader)
    obs_point = max(1, batch_len // 2)

    logger.info('Training started: batch_size={}, batch_len={}'.format(batch_size, batch_len))
    model.train()

    total_cnt = max_epoch * len(data_loader.dataset)
    process_bar = tqdm.tqdm(total=total_cnt, colour='yellow', ncols=120, unit_scale=True, desc="Training")

    max_norm = 1000 / max_epoch

    for i in range(max_epoch):
        if model.is_being_stoped:
            break

        for idx, (input, target) in enumerate(data_loader):
            optimizer.zero_grad()

            output, (hn, cn) = model(input)

            flat_output = output.view(-1, model.vocab_size)
            flat_target = target.view(-1)

            loss = loss_func(flat_output, flat_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            if is_visible:
                batch_idx_list.append((batch_len * i + idx) / batch_len)
                loss_list.append(loss.item())
                if len(batch_idx_list) >= win_size * 5:
                    batch_idx_list = batch_idx_list[-win_size:]
                    loss_list = loss_list[-win_size:]

            if idx % obs_point == 0:
                logger.debug('epoch={}, idx={}, loss={}'.format(i, idx, loss))

                mater = ('床前明月光', {}, '楊花落盡子規啼，聞道龍標過五溪。')
                poem, fix_num = generate_poem(model=model, device=device, dataset=dataset,
                                              starting=mater[0], acrostic=mater[1], conception=mater[2],
                                              poem_len=POEM_CHAR*4, topk=1)
                model.train()

                fix_tag = '[fixed-{}]'.format(fix_num) if fix_num > 0 else ''
                logger.debug('{} {}'.format(poem, fix_tag))

            process_bar.update(len(target))

    model.is_trained = True
    process_bar.close()

    if is_visible:
        vis_thread.join()

    end_time = time.time()
    logger.info('Training ended. Time used: {}'.format(human_readable_time(end_time - start_time)))

def generate_poem(model, dataset:TextDataSet, device:torch.device,
                  starting:str='', acrostic:dict[int:str]={}, conception:str='',
                  poem_len:int=8*4, topk=1):
    model.eval()

    if conception:
        conception_word = ['<P_BEGIN>'] + list(conception)
    else:
        conception_word = ['<P_BEGIN>']

    hn, cn = None, None

    if len(conception_word) > 1:
        conception_idx = [dataset.trans_word(word) for word in conception_word[:-1]]
        conception_idx_t = torch.tensor(conception_idx).view(1,-1).to(device) 
        output, (hn, cn) = model(conception_idx_t, (hn, cn))

    word_idx = dataset.trans_word(conception_word[-1])

    result_idx = []
    for i, s in enumerate(starting):
        acrostic[i+1] = s

    fix_num = 0

    for i in range(poem_len):
        word_idx_t = torch.tensor([word_idx]).view(1,1).to(device)
        output, (hn, cn) = model(word_idx_t, (hn, cn))

        if (i+1) in acrostic.keys():
            word_idx = dataset.trans_word(acrostic[i+1])
        else:
            if topk > 1:
                d = int(np.floor(np.abs(np.random.normal(0, topk/3))))
                d = min(d, dataset.get_vocab_size())
                d = min(d, 2*topk)
            else:
                d = 0
            word_idx = output.data[0][0].topk(2*topk)[1][d].item()

            unexpected_idx = set([dataset.trans_word(word) for word in ['<P_BEGIN>', '<P_END>', '<NULL>']])
            if i < poem_len:
                nd = -1
                while word_idx in unexpected_idx:
                    nd += 1
                    word_idx = output.data[0][0].topk(dataset.get_vocab_size())[1][nd].item()
                    fix_num += 1

        result_idx.append(word_idx)

    result = dataset.trans_index_list(result_idx)

    end = result.find('<P_END>')
    if end > -1:
        result = result[:end]

    return result, fix_num

if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_path = 'models/tang-7c-poem-lstm'

    # dataset = TextDataSet(type=TextType.TangPoem, pkl_name='tang-7c.pkl',
    #                       authors=None,
    #                       max_len=None, char_size=[POEM_CHAR*4, POEM_CHAR*8])
    # model = PoemLstmModel(num_classes=dataset.get_vocab_size(), emb_size=128,
    #                       hidden_size=512, num_layers=2, is_bidirectional=False, lstm_dropout=0.0,
    #                       fc_dropout=0.05, device=device)
    # train(model=model, dataset=dataset, max_epoch=200, learning_rate=0.001, batch_size=128,
    #       padding_len=POEM_CHAR*8+2+4, is_visible=True, win_size=1000, device=device)

    # torch.save(model, model_path)

    model = torch.load(model_path)
    model.move(device)
    dataset = TextDataSet(type=model.text_type, pkl_name=model.text_pkl_name)

    # materials_5c = [
    #     ('床前明月光，', {}, '霜，月，故鄉。'),
    #     ('故國三千里，', {}, '深宮，淚水。'),
    #     ('青山橫北郭，', {}, '白水，浮雲，故人。'),
    #     ('田家少閑月，', {}, '小麥，農民，飢腸，悲傷。'),
    # ]

    materials_7c = [
        ('深宮秋月幾許愁，', {1:'深', 9:'度', 17:'學', 25:'習'}, '疑是銀河落九天。'),
        ('名花傾國兩相歡，', {}, '馬嵬坡，戰亂，生離死別，君王，紅顔，悲情。'),
        ('楊花落盡子規啼，', {}, '長江，天際，星空。'),
        ('飛流直下三千尺，', {}, '銀漢，星空，疾風，九天，明月。'),
        ('朝辭白帝彩雲間，', {}, '李白斗酒詩百篇。'),
    ]
    poem_len = POEM_CHAR * 4

    logger.info('Testing in creating new poems...')
    logger.info('')

    for mater in materials_7c:
        poem, fix_num = generate_poem(model=model, device=device, dataset=dataset,
                                      starting=mater[0], acrostic=mater[1], conception=mater[2],
                                      poem_len=poem_len, topk=1)

        fix_tag = '[fixed-{}]'.format(fix_num) if fix_num > 0 else ''
        logger.info('{} {}'.format(poem, fix_tag))
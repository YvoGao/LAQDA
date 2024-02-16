
import csv
import os
import torch
import json
import numpy as np
from tqdm import tqdm
from utils import get_parser
import random
from losses import Loss_fn
from encoder import MyModel

from transformers import AdamW, get_linear_schedule_with_warmup

from data_loader import MyDataset, KShotTaskSampler, get_label_dict
from collections import defaultdict
# from tensorboardX import SummaryWriter


def set_seed(seed):
    """
        Setting random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def init_dataloader(args, mode):
    filePath = os.path.join(args.dataFile, mode + '.json')
    if mode == 'train' or mode == 'valid':
        episode_per_epoch = args.episodeTrain
    else:
        episode_per_epoch = args.episodeTest
    dataset = MyDataset(filePath)
    sampler = KShotTaskSampler(dataset, episodes_per_epoch=episode_per_epoch, n=args.numKShot, k=args.numNWay, q=args.numQShot, num_tasks=1)

    return sampler


def save_list_to_file(path, thelist):
    with open(path, 'a+') as f:
        for item in thelist:
            f.write("%s\n" % item)


def init_model(args):
    device = torch.device('cuda', args.numDevice)
    torch.cuda.set_device(device)
    model = MyModel(args).to(device)
    return model

def init_optim(args, model):

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    return optimizer

def init_lr_scheduler(args, optim):
    '''
    Initialize the learning rate scheduler
    '''
    
    t_total = args.epochs * args.episodeTrain
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    return scheduler


def deal_data(support_set, query_set, episode_labels):
    '''
    change label to one-hot
    '''
    text, labels = [], []
    for x in support_set:
        text.append(x["text"])
        labels.append(x["label"])
    for x in query_set:
        text.append(x["text"])
        labels.append(x["label"])  
    label_ids = []
    for label in labels:
        tmp = []
        for l in episode_labels:
            if l == label:
                tmp.append(1)
            else:
                tmp.append(0)
        label_ids.append(tmp)

    return text, label_ids

def train(args, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):

    # train_writer = SummaryWriter(os.path.join(args.fileModelSave, 'train_log'))
    
    if val_dataloader is None:
        acc_best_state = None
    
    train_loss, epoch_train_loss = [], []
    train_acc, epoch_train_acc = [], []
    train_p, epoch_train_p = [], []
    train_r, epoch_train_r = [], []
    train_f1, epoch_train_f1 = [], []
    train_auc, epoch_train_auc = [], []
    train_topkacc, epoch_train_topkacc = [], []
    val_loss, epoch_val_loss = [], []
    val_acc, epoch_val_acc = [], []
    val_p, epoch_val_p = [], []
    val_r, epoch_val_r = [], []
    val_f1, epoch_val_f1 = [], []
    val_auc, epoch_val_auc = [], []
    val_topkacc, epoch_val_topkacc = [], []
    best_p = 0
    best_r = 0
    best_f1 = 0
    best_acc = 0
    best_auc = 0
    loss_fn = Loss_fn(args)
    
    acc_best_model_path = os.path.join(args.fileModelSave, 'acc_best_model.pth')
    cycle = 0
    labels_dict = get_label_dict(args)
    id2label = {}
    for x, y in labels_dict.items():
        id2label[y] = x 
    for epoch in range(args.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        model.train()
        if cycle == args.patience:
            break
        for  i, batch in tqdm(enumerate(tr_dataloader)):
            optim.zero_grad()
            support_set, query_set, episode_labels = batch
            # 这个地方可以得到labels
            # import pdb
            # pdb.set_trace()
            label_text = [id2label[int(el)] for el in episode_labels]
            text, labels = deal_data(support_set, query_set, episode_labels)
           
            model_outputs = model(text, label_text)
           
            loss, p, r, f, acc, auc, topk_acc= loss_fn(model_outputs, labels)
            
            loss.backward()
            optim.step()
            lr_scheduler.step()
            train_loss.append(loss.item())
            train_p.append(p)
            train_r.append(r)
            train_f1.append(f)
            train_acc.append(acc)
            train_auc.append(auc)
            train_topkacc.append(topk_acc)
            print('Train Loss: {}, Train p: {}, Train r: {}, Train f1: {},  Train acc: {},  Train auc: {}, Train topk acc: {}'.format(loss, p, r, f, acc, auc, topk_acc))

        avg_loss = np.mean(train_loss[-args.episodeTrain:])
        avg_acc = np.mean(train_acc[-args.episodeTrain:])
        avg_p = np.mean(train_p[-args.episodeTrain:])
        avg_r = np.mean(train_r[-args.episodeTrain:])
        avg_f1 = np.mean(train_f1[-args.episodeTrain:])
        avg_auc = np.mean(train_auc[-args.episodeTrain:])
        avg_topkacc = np.mean(train_topkacc[-args.episodeTrain:])
        print('Avg Train Loss: {}, Avg Train p: {}, Avg Train r: {}, Avg Train f1: {}, Avg Train acc: {}, Avg Train auc: {}, Avg Train topk acc: {}'.format(avg_loss, avg_p, avg_r, avg_f1, avg_acc, avg_auc, avg_topkacc))
        epoch_train_loss.append(avg_loss)
        epoch_train_acc.append(avg_acc)
        epoch_train_p.append(avg_p)
        epoch_train_r.append(avg_r)
        epoch_train_f1.append(avg_f1)
        epoch_train_auc.append(avg_auc)
        epoch_train_topkacc.append(avg_topkacc)

        if val_dataloader is None:
            continue
        with torch.no_grad():
            model.eval()
            
            for batch in tqdm(val_dataloader):
                support_set, query_set, episode_labels = batch
                text, labels = deal_data(support_set, query_set, episode_labels)
                label_text = [id2label[int(el)] for el in episode_labels]
                model_outputs = model(text, label_text)
                loss, p, r, f, acc, auc, topkacc = loss_fn(model_outputs, labels)
                
                val_loss.append(loss.item())
                val_acc.append(acc)
                val_p.append(p)
                val_r.append(r)
                val_f1.append(f)
                val_auc.append(auc)
                val_topkacc.append(topkacc)
                
            avg_loss = np.mean(val_loss[-args.episodeTrain:])
            avg_acc = np.mean(val_acc[-args.episodeTrain:])
            avg_p = np.mean(val_p[-args.episodeTrain:])
            avg_r = np.mean(val_r[-args.episodeTrain:])
            avg_f1 = np.mean(val_f1[-args.episodeTrain:])
            avg_auc = np.mean(val_auc[-args.episodeTrain:])
            avg_topkacc = np.mean(val_topkacc[-args.episodeTrain:])
            epoch_val_loss.append(avg_loss)
            epoch_val_acc.append(avg_acc)
            epoch_val_p.append(avg_p)
            epoch_val_r.append(avg_r)
            epoch_val_f1.append(avg_f1)
            epoch_val_auc.append(avg_auc)
            epoch_val_topkacc.append(avg_topkacc)

        postfix = ' (Best)' if avg_p >= best_p else ' (Best: {})'.format(
            best_p)
        r_prefix = ' (Best)' if avg_r >= best_r else ' (Best: {})'.format(
            best_r)
        f1_prefix = ' (Best)' if avg_f1 >= best_f1 else ' (Best: {})'.format(
            best_f1)
        acc_prefix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        auc_prefix = ' (Best)' if avg_auc >= best_auc else ' (Best: {})'.format(
            best_auc)
        print('Avg Val Loss: {}, Avg Val p: {}{}, Avg Val r: {}{}, Avg Val f1: {}{}, Avg Val acc: {}{}, Avg Val auc: {}{},  Avg Val topkacc: {}'.format(
            avg_loss, avg_p, postfix, avg_r, r_prefix, avg_f1, f1_prefix, avg_acc, acc_prefix, avg_auc, auc_prefix, avg_topkacc))
   
       
        cycle +=1
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), acc_best_model_path)
            best_acc = avg_acc
            acc_best_state = model.state_dict()
            cycle = 0
        
       
    
    for i, t_f in enumerate(train_f1):
        train_writer.add_scalar("Train/F1", t_f, i)
        train_writer.add_scalar("Train/Loss", train_loss[i], i)

    for i, t_f in enumerate(val_f1):
        train_writer.add_scalar("Val/F1", t_f, i)
        train_writer.add_scalar("Val/Loss", val_loss[i], i)

    for name in ['epoch_train_loss', 'epoch_train_p', 'epoch_train_r', 'epoch_train_f1', 'epoch_train_acc', 'epoch_train_auc', 'epoch_train_topkacc', 'epoch_val_loss', 'epoch_val_p', 'epoch_val_r', 'epoch_val_f1', 'epoch_val_acc', 'epoch_val_auc', 'epoch_val_topkacc']:
        save_list_to_file(os.path.join(args.fileModelSave,
                                       name + '.txt'), locals()[name])

    return model
        

def test(args, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    val_p = []
    val_r = []
    val_loss = []
    val_f1 = []
    val_acc = []
    val_auc = []
    val_topkacc = []
    loss_fn = Loss_fn(args)
    with torch.no_grad():
        model.eval()
        labels_dict = get_label_dict(args)
        id2label = {}
        for x, y in labels_dict.items():
            id2label[y] = x 
        for batch in tqdm(test_dataloader):
            support_set, query_set, episode_labels = batch
            text, labels = deal_data(support_set, query_set, episode_labels)
            label_text = [id2label[int(el)] for el in episode_labels]
            model_outputs = model(text, label_text)
            loss, p, r, f, acc, auc, topkacc = loss_fn(model_outputs, labels)
            
            val_loss.append(loss.item())
            val_acc.append(acc)
            val_p.append(p)
            val_r.append(r)
            val_f1.append(f)
            val_auc.append(auc)
            val_topkacc.append(topkacc)
                
        avg_loss = np.mean(val_loss)
        avg_acc = np.mean(val_acc)
        avg_p = np.mean(val_p)
        avg_r = np.mean(val_r)
        avg_f1 = np.mean(val_f1)
        avg_auc = np.mean(val_auc)
        avg_topkacc = np.mean(val_topkacc)


        print('Test p: {}'.format(avg_p))
        print('Test r: {}'.format(avg_r))
        print('Test f1: {}'.format(avg_f1))
        print('Test acc: {}'.format(avg_acc))
        print('Test auc: {}'.format(avg_auc))
        print('Test topkacc: {}'.format(avg_topkacc))
        print('Test Loss: {}'.format(avg_loss))

        path = args.fileModelSave + "/test_score.json"
        with open(args.fileModelSave+'/result.csv', 'a+', newline="") as f:
            writer = csv.writer(f)
            # writer.writerow(colmuns)
            data = ["commont", args.commont, "data", args.dataFile,"shot", args.numKShot, "acc", avg_acc]
            writer.writerow(data)
        with open(path, "a+") as fout:
            tmp = {"commont": args.commont, "data":args.dataFile,"shot": args.numKShot, "acc": avg_acc, "p": avg_p, "r": avg_r, "f1": avg_f1, "auc": avg_auc, "Loss": avg_loss}
            fout.write("%s\n" % json.dumps(tmp, ensure_ascii=False))



def write_args_to_josn(args):
    path = args.fileModelSave + "/config.json"
    args = vars(args)
    json_str = json.dumps(args, indent=4)
    with open(path, 'a+') as json_file:
        json_file.write(json_str)
        

def main():
    args = get_parser().parse_args()

    if not os.path.exists(args.fileModelSave):
        os.makedirs(args.fileModelSave)

    write_args_to_josn(args)
    set_seed(42)
    model = init_model(args)
    # print(model)

    tr_dataloader = init_dataloader(args, 'train')
    val_dataloader = init_dataloader(args, 'valid')
    test_dataloader = init_dataloader(args, 'test')

    optim = init_optim(args, model)
    lr_scheduler = init_lr_scheduler(args, optim)
    model = train(args=args,
                    tr_dataloader=tr_dataloader,
                    val_dataloader=val_dataloader,
                    model=model,
                    optim=optim,
                    lr_scheduler=lr_scheduler)
    
    # print('Testing with last model..')
    # test(args=args,
    #      test_dataloader=test_dataloader,
    #      model=model)
    
    model.load_state_dict(torch.load(args.fileModelSave + "/acc_best_model.pth"))
    print('Testing with acc best model..')
    test(args=args,
         test_dataloader=test_dataloader,
         model=model)

   
if __name__ == '__main__':
    main()

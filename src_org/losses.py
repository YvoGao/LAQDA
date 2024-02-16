import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn.functional as F
from sklearn.metrics import precision_score,f1_score,recall_score, accuracy_score, roc_auc_score


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class Loss_fn(torch.nn.Module):
    def __init__(self, args):
        super(Loss_fn, self).__init__()
        self.args = args

    def forward(self, model_outputs, labels):
        query_size = self.args.numNWay * self.args.numQShot
        support_size = self.args.numNWay * self.args.numKShot

        prototypes, q_re, topk_acc, original_prototypes, sampled_data = model_outputs

        dists = euclidean_dist(q_re, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1) # num_query x num_class
     
        query_labels = labels[support_size:]
        query_labels = torch.tensor(query_labels, dtype=float).cuda()
        loss = - query_labels * log_p_y
        loss = loss.mean()

        # calculate generate loss:
        generate_labels = torch.tensor(range(self.args.numNWay)).unsqueeze(dim=1)
        generate_labels = generate_labels.repeat(1, sampled_data.shape[1]).view(-1)
        generate_labels = F.one_hot(generate_labels, self.args.numNWay).float().cuda()
        sampled_data = sampled_data.view(-1, sampled_data.shape[2])
        g_dists = euclidean_dist(sampled_data, original_prototypes)
        glog_p_y = F.log_softmax(-g_dists, dim=1)
        g_loss = - generate_labels * glog_p_y
        g_loss= g_loss.mean()

        # 交叉熵原来的+生成的
        overall_loss = loss + 0.1 * g_loss

        
        # single:
        x, _ = torch.max(log_p_y, dim=1, keepdim=True)
        one = torch.ones_like(log_p_y)
        zero = torch.zeros_like(log_p_y)
        y_pred = torch.where(log_p_y >= x, one, log_p_y)
        y_pred = torch.where(y_pred < x, zero, y_pred)

        # print(y_pred)
        target_mode = 'macro'

        query_labels = query_labels.cpu().detach()
        y_pred = y_pred.cpu().detach()
        p = precision_score(query_labels, y_pred, average=target_mode)
        r = recall_score(query_labels, y_pred, average=target_mode)
        f = f1_score(query_labels, y_pred, average=target_mode)
        acc = accuracy_score(query_labels, y_pred)

        y_score = F.softmax(-dists, dim=1)
        y_score = y_score.cpu().detach()
        auc = roc_auc_score(query_labels, y_score)

       
        return overall_loss, p, r, f, acc, auc, topk_acc

       







import torch
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F
import model
import Pool
from DataLoader import MyDataset
import torch.nn.functional as F
from torch import optim
from Buffer import Buffer
import numpy as np
#import mc_eced
#import mc_eced_mini
#import mc_eced_slow
#import mc_eced_pre_gpu
import mc_eced_gpu
import sampler_model

sample_num=100
al_num=6
#hamming_dis_threshold=0.05
patience=3
dropout_rate=0.5
B = 10
M = 10000
sampling_index=2

batch_size=64
num_workers=5

#hamming_pool_size = 10000
#val_dataset_size = 10000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_eval_model(dataloader_train, dataloader_val, dataloader_test, weight):
    net = model.BayesianNet(10).to(device)
    opt = optim.Adam(list(net.parameters()), lr=0.001)
    
    train_net = sampler_model.SamplerModel(net, k=1)
    val_net = sampler_model.NoDropoutModel(net)
    test_net = sampler_model.SamplerModel(net, k=sample_num)

    best_acc = 0.0
    best_epoch = 0
    best_model_state = copy.deepcopy(net.state_dict())
    weight = torch.Tensor(weight).to(device)
     
    for epoch in range(500):
        net.train()
        for b_id, (x, label) in enumerate(dataloader_train):
            x, label = x.to(device), label.to(device)
            opt.zero_grad()
            pred = train_net(x)
            label = label.view(-1)
            loss = F.nll_loss(pred, label, weight=weight)
            loss.backward()
            opt.step()
        
        net.eval()
        ground_truth_lt = []
        pred_lt = []
        with torch.no_grad():
            for b_id, (x, label) in enumerate(dataloader_val):
                x, label = x.to(device), label.to(device)
                pred = val_net(x)
                label = label.view(-1)
                pred_lt += list(pred.data.cpu().numpy())
                ground_truth_lt += list(label.data.cpu().numpy())

        pred_lt = np.array(pred_lt)
        ground_truth_lt = np.array(ground_truth_lt)

        pred_index = np.argmax(pred_lt, axis=1)
        acc = np.sum(pred_index == ground_truth_lt)/len(ground_truth_lt)
        #print(f'... val acc: {acc}')
        if acc <= best_acc:
            if epoch - best_epoch > patience :
                print(f'patience ex; best epoch: {best_epoch}, best val acc: {best_acc}')
                break
        else:
            best_acc = acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(net.state_dict())

    net.load_state_dict(best_model_state)
    net.eval()
    ground_truth_lt = []
    pred_lt = []
    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader_test):
            x, label = x.to(device), label.to(device)
            pred = test_net(x)
            label = label.view(-1)
            pred_lt += list(pred.data.cpu().numpy())
            ground_truth_lt += list(label.data.cpu().numpy())

    pred_lt = np.array(pred_lt)
    ground_truth_lt = np.array(ground_truth_lt)

    pred_index = np.argmax(pred_lt, axis=1)
    test_acc = np.sum(pred_index == ground_truth_lt)/len(ground_truth_lt)
    
    return net, best_acc, test_acc

def one_pass(pass_time):
    buff_val = Buffer()
    #buff_test = Buffer()
    buff_train = Buffer()
    buff_hamming = Buffer()

    pool = Pool.Pool('./data/MNIST/fixed/training.pt')
    pool_test = Pool.Pool('./data/MNIST/fixed/test.pt')

    #feat_lt, label_lt = pool.random_query_labels(val_dataset_size)
    #buff_val.add_rows(feat_lt, label_lt)
    buff_val.add_from_pt('./data/MNIST/fixed/val.pt')

    #feat_lt, label_lt = pool.random_query_balanced_labels(K=2)
    #buff_train.add_rows(feat_lt, label_lt)
    buff_train.add_from_pt('./data/MNIST/fixed/start.pt')

    #feat_lt, label_lt = pool.random_query_labels(hamming_pool_size)
    #buff_hamming.add_rows(feat_lt, label_lt)
    buff_hamming.add_from_pt('./data/MNIST/fixed/hamming.pt')

    dataset_val = MyDataset(buff_val)
    dataloader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            )

    dataset_test = MyDataset(pool_test)
    dataloader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            )

    test_acc_lt = []
    while len(buff_train.labels) < 300:
        dataset_train = MyDataset(buff_train)
        weight = dataset_train.get_weight()
        if len(buff_train.labels) < 300:
            sampler = torch.utils.data.RandomSampler(dataset_train, replacement=True, num_samples=4096)
            dataloader_train = torch.utils.data.DataLoader(
                    dataset_train,
                    sampler=sampler,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    #shuffle=True,
                )
        else:
            dataloader_train = torch.utils.data.DataLoader(
                    dataset_train,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    drop_last=True
                )


        net, val_acc, test_acc = train_eval_model(dataloader_train, dataloader_val, dataloader_test, weight=weight)
        hamming_dis_threshold=(1.0-val_acc)/4
        _, pos_lt = mc_eced_gpu.delta_batch_mc_eced(net, pool, buff_hamming, M=M, B=B, sampling_index=sampling_index, device=device, sample_num=sample_num, hamming_dis_threshold=hamming_dis_threshold)
        feat_lt, label_lt = pool.query_labels(pos_lt)
        buff_train.add_rows(feat_lt, label_lt)
        print(f'ECED MC Batch Pass:{pass_time}, train_size:{len(dataset_train.labels)}, test_acc:{test_acc}, pos_lt: {pos_lt} ')
        test_acc_lt.append(test_acc)

    return test_acc_lt

acc_lts = []
for pass_time in range(al_num):
    test_acc_lt = one_pass(pass_time)
    acc_lts.append(test_acc_lt)

    acc_lts_np = np.array(acc_lts)
    np.savetxt(f'ECED-MC-Batch_sample-num-{sample_num}_hamming-anneal-div4_B-{B}_M-{M}_patience-{patience}.out', acc_lts_np, delimiter=',')

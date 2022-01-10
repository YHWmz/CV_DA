import configargparse
import data_loader
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random
from scipy.spatial.distance import cdist
from torch import nn
import copy
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data_loader import *
from sklearn.neighbors import NeighborhoodComponentsAnalysis,KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)
    
    # training related
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=60, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False, help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=10)
    parser.add_argument('--transfer_loss', type=str, default='mmd')
    parser.add_argument('--is_balance', type=str2bool, default=False)
    return parser

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    source_loader, n_class = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=True, train=True, num_workers=args.num_workers)
    target_train_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=False, train=True, num_workers=args.num_workers)
    target_test_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    return source_loader, target_train_loader, target_test_loader, n_class

def get_model(args):
    #model = models.TransferNet(
    #    args.n_class, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck).to(args.device)
    model = models.plabel_coorATT_knn(args.n_class, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter,
        use_bottleneck=args.use_bottleneck).to(args.device)

    return model

def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler

def test(model, target_test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = 100. * correct / len_target_dataset
    return acc, test_loss.avg

def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args):
    print('balance para is :', args.is_balance)
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)

    n_batch = len_target_loader
    print('batch_num is ', n_batch)

    if n_batch == 0:
        n_batch = args.n_iter_per_epoch 
    
    iter_source = iter(source_loader)

    best_acc = 0
    stop = 0
    log = []

    # new_target_loader = obtain_label(target_train_loader, model, args)
    # if True:
    #     return 0
    pflag = 0
    for e in range(0, args.n_epoch+1):
        if e == 1000:
            if args.src_domain=='Clipart':
                model.load_state_dict(torch.load('/DB/rhome/yuhaowang/CV_DA/transferlearning-master/code/DeepDA/C2Rmodel30.pth'))
                #model.load_state_dict(torch.load('/DB/rhome/yuhaowang/CV_DA/transferlearning-master/code/DeepDA/plabel_Att_knn/tenepoch.pth'))
                print('load success')
            elif args.src_domain=='Art':
                model.load_state_dict(torch.load('/DB/rhome/yuhaowang/CV_DA/transferlearning-master/code/DeepDA/A2Rmodel30.pth'))
                print('load success')
            else:
                model.load_state_dict(torch.load('/DB/rhome/yuhaowang/CV_DA/transferlearning-master/code/DeepDA/P2Rmodel30.pth'))
                print('load success')

        #     torch.save(model.state_dict(),'/DB/rhome/yuhaowang/CV_DA/transferlearning-master/code/DeepDA/plabel_Att_knn/tenepoch.pth')
        #     print('save success')
        #print(pflag)
        if e >= 0 and e % 10 == 0:
            pflag = 1
            new_target_loader = obtain_soft_label(target_train_loader, model, args, e, source_loader)
        else:
            if pflag == 0:
                new_target_loader = target_train_loader
            else:
                pflag -= 1

        model.train()
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch)

        if max(len_target_loader, len_source_loader) != 0:
            iter_source = iter(source_loader)

        criterion = torch.nn.CrossEntropyLoss()
        #for ii in range(n_batch):
        for ii, (data_target, plabel) in enumerate(new_target_loader):
            #data_source, label_source = next(iter_source) # .next()
            #data_source, label_source = data_source.to(args.device), label_source.to(args.device)
            data_target = data_target.to(args.device)
            plabel = plabel.to(args.device)
            if pflag == 0:
                plabel = None

            #clf_loss, clf_loss_t, transfer_loss = model(data_source, data_target, label_source, plabel)
            clf_loss_t, transfer_loss = model.forward_stage2(data_target, plabel)

            if clf_loss_t == 0:
                loss = args.transfer_loss_weight * transfer_loss
            else:
                loss = args.transfer_loss_weight * transfer_loss + clf_loss_t
            # if clf_loss_t == 0:
            #     loss = clf_loss + args.transfer_loss_weight * transfer_loss
            # else:
            #     loss = clf_loss*0.5 + 0*args.transfer_loss_weight * transfer_loss + clf_loss_t
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_clf.update(0)
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])
        
        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
                        e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)
        # Test
        stop += 1
        test_acc, test_loss = test(model, target_test_loader, args)
        info += ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        if best_acc < test_acc:
            #torch.save(model.state_dict(),'/DB/rhome/yuhaowang/CV_DA/transferlearning-master/code/DeepDA/plabel_Att_knn/tenepoch.pth')
            #print('save success')
            best_acc = test_acc
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        print(info)
        if e%20 == 0:
            print('Transfer result: {:.4f}'.format(best_acc))
    print('Transfer result: {:.4f}'.format(best_acc))

def obtain_label(loader, model, args, e):
    # model.base_network, model.bottleneck_layer, model.classifier_layer
    # netF, netB, netC, args
    start_test = True
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.cuda()

            features_map = model.base_network(inputs)
            features = model.base_network.avgpool(features_map).view(features_map.size(0), -1)

            feas = model.bottleneck_layer(features)
            outputs = model.classifier_layer(feas)

            if start_test:
                all_inputs = inputs.float().cpu()
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_inputs = torch.cat((all_inputs, inputs.float().cpu()), 0)
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
    unknown_weight = 1 - ent / np.log(all_output.shape[1])
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if True:
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    # 初始化中心？
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    # 预测结果中每种class的个数
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>0)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], 'cosine')
    pred_label = dd.argmin(axis=1)

    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
        top2_idx = torch.topk(torch.tensor(-dd),4,1)[1]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    print(log_str+'\n')

    pred_label = torch.tensor(pred_label.astype('int'))

    if args.is_balance:
        new_dataset = MyClassBalanceDataset(all_inputs, pred_label.reshape(-1, 1))
        _batchSampler = MyBatchSampler(new_dataset, 32)
        new_dataloader = DataLoader(dataset=new_dataset,num_workers=2,batch_sampler=_batchSampler)
    else:
        new_dataset = TensorDataset(all_inputs, pred_label.reshape(-1,1))
        new_dataloader = DataLoader(dataset=new_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    return new_dataloader

def obtain_soft_label(loader, model, args, e, src_loader):
    # model.base_network, model.bottleneck_layer, model.classifier_layer
    # netF, netB, netC, args
    start_test = True
    with torch.no_grad():
        for (inputs, labels), (src_in, src_lab) in zip(loader, src_loader):
            inputs = inputs.cuda()
            src_in = src_in.cuda()

            features_map = model.base_network(inputs)
            features = model.base_network.avgpool(features_map).view(features_map.size(0), -1)
            src_features_map = model.base_network(src_in)
            src_features = model.base_network.avgpool(src_features_map).view(features_map.size(0), -1)

            feas = model.bottleneck_layer(features)
            outputs = model.classifier_layer(feas)

            src_feas = model.bottleneck_layer(src_features)

            if start_test:
                all_inputs = inputs.float().cpu()
                all_fea = feas.float().cpu()
                src_all_fea = src_feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                src_all_label = src_lab.float()
                start_test = False
            else:
                all_inputs = torch.cat((all_inputs, inputs.float().cpu()), 0)
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                src_all_fea = torch.cat((src_all_fea, src_feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                src_all_label = torch.cat((src_all_label, src_lab.float()), 0)

    ####################################################################################################################
    print('begin tsne')
    color = ['grey', 'gold','darkviolet','turquoise','r','g','b','c','m','y','k','darkorange','lightgreen','plum','tan','khaki','pink','skyblue','lawngreen','salmon']

    print(all_fea.numpy().shape, type(all_fea.numpy()), all_label.numpy().shape)
    plot_fea = np.concatenate([all_fea.numpy(), src_all_fea.numpy()])
    plot_label = np.concatenate([all_label,src_all_label]).astype(np.int32)
    size = all_label.shape[0]
    X_embedded = TSNE(n_components=2, metric='cosine').fit_transform(plot_fea)
    #X_label = all_label.numpy().astype(np.int32)
    print(X_embedded.shape, plot_label.shape)
    for i,key in enumerate(mcd.CSS4_COLORS):
        if i == 20:
            break
        idxt = (plot_label==i+20)
        idxt[size:] = 0
        # mcd.CSS4_COLORS[key]
        plt.scatter(X_embedded[idxt][0:20,0], X_embedded[idxt][0:20,1], c=color[i], s=20,marker='.')
        idxs = (plot_label == i)
        idxs[0:size] = 0
        plt.scatter(X_embedded[idxs][0:20, 0], X_embedded[idxs][0:20, 1], c=color[i], s=20, marker='2')

    plt.savefig('allfea.png')

    ####################################################################################################################

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
    unknown_weight = 1 - ent / np.log(all_output.shape[1])
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if True:
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    # 初始化中心？
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    # 预测结果中每种class的个数
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>0)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], 'cosine')
    pred_label = dd.argmin(axis=1)

    first_dis = dd.min(axis=1)
    dd_sec = copy.deepcopy(dd)
    for i in range(dd.shape[0]):
        dd_sec[i, pred_label[i]] = 1e5
    sec_dis = dd_sec.min(axis=1)
    ratio = first_dis/sec_dis
    threhold = 0.6
    print(np.sum(ratio > threhold)/ratio.shape[0])

    negative_sample = np.where(ratio>threhold)
    positive_sample = np.where(ratio<=threhold)

    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
        # acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
        # log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
        #
        # print(log_str + '\n')
        top2_idx = torch.topk(torch.tensor(-dd),4,1)[1]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    print(log_str+'\n')

    # print('*'*20)
    # print('*'*5+'    start knn!    ' + '*'*5)
    # print('*'*20)
    # x_train = all_fea[positive_sample]
    # y_train = pred_label[positive_sample]
    #
    # knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    # knn.fit(x_train, y_train)
    # # nca = NeighborhoodComponentsAnalysis(random_state=42)
    # # knn = KNeighborsClassifier(n_neighbors=5)
    # # nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
    # # nca_pipe.fit(x_train, y_train)
    # print('*'*5+'    training finish! Start to test    ' + '*'*5)
    #
    # x_test = all_fea[negative_sample]
    # y_test = all_label[negative_sample].float().numpy()
    # pred_negative = knn.predict(x_test)
    #
    # # print(np.sum(pred_label == all_label.float().numpy()))
    # # print(np.sum(pred_label[negative_sample] == all_label[negative_sample].float().numpy()))
    # # print(np.sum(y_train == all_label[positive_sample].float().numpy()))
    # # print(len(y_test))
    # # print(len(y_train))
    # # print(np.sum(y_test == pred_label[negative_sample]) / len(y_test))
    # # print(np.sum(y_test == pred_negative) / len(y_test))
    # log_str_knn = 'Accuracy = {:.2f}% -> {:.2f}%'.format(np.sum(y_test == pred_label[negative_sample]) / len(y_test) * 100, np.sum(y_test == pred_negative) / len(y_test) * 100)
    # print(log_str_knn + '\n')
    # # print(np.sum(y_train == all_label[positive_sample].float().numpy())/ len(y_train))
    # # #print(np.sum(all_label[positive_sample].float().numpy() == pred_positive) / len(y_train))
    #
    # knnacc = (np.sum(pred_negative == y_test)+ np.sum(y_train==all_label[positive_sample].float().numpy()))/ len(all_fea)
    # log_str_knn = 'Accuracy = {:.2f}% -> {:.2f}%'.format(acc * 100, knnacc * 100)
    # print(log_str_knn + '\n')

    # acc1 = np.sum(pred_label[negative_sample] == all_label[negative_sample].float().numpy()) / len(
    #     pred_label[negative_sample])

    pred_label1 = copy.deepcopy(pred_label)
    pred_label2 = copy.deepcopy(pred_label)

    pred_label1[negative_sample] = top2_idx[negative_sample, 1].numpy()
    pred_label2[negative_sample] = top2_idx[negative_sample, 2].numpy()
    pred_label = np.concatenate((pred_label,pred_label1,pred_label2))

    all_inputs = torch.cat([all_inputs, all_inputs, all_inputs],0)
    gt = np.concatenate((all_label.float().numpy(),all_label.float().numpy(),all_label.float().numpy()),axis=0)
    print(np.sum(pred_label == gt) / len(all_inputs))
    pred_label = torch.tensor(pred_label.astype('int'))

    # acc2 = (np.sum(top2_idx[negative_sample,0].numpy() == all_label[negative_sample].float().numpy())+
    #         np.sum(top2_idx[negative_sample,1].numpy() == all_label[negative_sample].float().numpy())+
    #         np.sum(top2_idx[negative_sample,2].numpy() == all_label[negative_sample].float().numpy())+
    #         np.sum(top2_idx[negative_sample,3].numpy() == all_label[negative_sample].float().numpy())
    #         ) / len(all_fea[negative_sample])
    # log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(acc1 * 100, acc2 * 100)
    #
    # print(log_str + '\n')

    if args.is_balance:
        new_dataset = MyClassBalanceDataset(all_inputs, pred_label.reshape(-1, 1))
        _batchSampler = MyBatchSampler(new_dataset, 32)
        new_dataloader = DataLoader(dataset=new_dataset,num_workers=2,batch_sampler=_batchSampler)
    else:
        new_dataset = TensorDataset(all_inputs, pred_label.reshape(-1,1))
        new_dataloader = DataLoader(dataset=new_dataset, batch_size=32, shuffle=True, num_workers=2)

    return new_dataloader


def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    set_random_seed(args.seed)
    source_loader, target_train_loader, target_test_loader, n_class = load_data(args)
    setattr(args, "n_class", n_class)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args)
    

if __name__ == "__main__":
    main()

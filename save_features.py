import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py

import backbone
from data.datamgr import SimpleDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file
from model_resnet import *

def save_features(model, data_loader, outfile ):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    for i, (x,y) in enumerate(data_loader):
        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
        all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()

if __name__ == '__main__':
    params = parse_args('save_features')

    isAircraft = (params.dataset == 'aircrafts')

    assert params.method != 'maml' and params.method != 'maml_approx', 'maml do not support save_feature and run'

    image_size = params.image_size
    loadfile = os.path.join('filelists', params.dataset, 'novel.json')

    if params.json_seed is not None:
        checkpoint_dir = '%s/checkpoints/%s_%s/%s_%s_%s' %(configs.save_dir, params.dataset, params.json_seed, params.date, params.model, params.method)
    else:
        checkpoint_dir = '%s/checkpoints/%s/%s_%s_%s' %(configs.save_dir, params.dataset, params.date, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++'] :
        checkpoint_dir += '_%dway_%dshot_%dquery' %( params.train_n_way, params.n_shot, params.n_query)

    ## Use another dataset (dataloader) for unlabeled data
    if params.dataset_unlabel is not None:
        checkpoint_dir += params.dataset_unlabel
        checkpoint_dir += str(params.bs)

    ## Add jigsaw
    if params.jigsaw:
        checkpoint_dir += '_jigsaw_lbda%.2f'%(params.lbda)
        checkpoint_dir += params.optimization
    ## Add rotation
    if params.rotation:
        checkpoint_dir += '_rotation_lbda%.2f'%(params.lbda)
        checkpoint_dir += params.optimization

    checkpoint_dir += '_lr%.4f'%(params.lr)
    if params.finetune:
        checkpoint_dir += '_finetune'

    print('checkpoint_dir:',checkpoint_dir)

    if params.loadfile != '':
        modelfile   = params.loadfile
        checkpoint_dir = params.loadfile
    else:
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
        elif params.method in ['baseline', 'baseline++'] :
            modelfile   = get_resume_file(checkpoint_dir)
        else:
            modelfile   = get_best_file(checkpoint_dir)

    if params.save_iter != -1:
        outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + "_" + str(params.save_iter)+ ".hdf5")
    else:
        outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + ".hdf5")

    datamgr         = SimpleDataManager(image_size, batch_size = params.test_bs, isAircraft=isAircraft)
    data_loader      = datamgr.get_data_loader(loadfile, aug = False)

    if params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4':
            model = backbone.Conv4NP()
        elif params.model == 'Conv6':
            model = backbone.Conv6NP()
        elif params.model == 'Conv4S':
            model = backbone.Conv4SNP()
        else:
            model = model_dict[params.model]( flatten = False )
    elif params.method in ['maml' , 'maml_approx']:
       raise ValueError('MAML do not support save feature')
    else:
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot, \
                                        jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation, tracking=params.tracking)
        if params.method == 'protonet':
            print("USE BN:",not params.no_bn)
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params , use_bn = (not params.no_bn))
        elif params.method == 'matchingnet':
            model           = MatchingNet( model_dict[params.model], **train_few_shot_params )
        else:# baseline and baseline++
            if isinstance(model_dict[params.model],str):
                if model_dict[params.model] == 'resnet18':
                    model = ResidualNet('ImageNet', 18, 1000, None)
            else:
                model = model_dict[params.model]()

    model = model.cuda()
    if params.method != 'baseline':
        model.feature = model.feature.cuda()

    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
            state[newkey] = state.pop(key)
        else:
            state.pop(key)

    model.feature.load_state_dict(state)
    model.feature.eval()
    model.eval()

    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    print('outfile is', outfile)
    save_features(model, data_loader, outfile)

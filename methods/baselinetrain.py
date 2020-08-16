import backbone
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from model_resnet import *
from resnet_pytorch import *

class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type = 'softmax', jigsaw=False, lbda=0.0, rotation=False, tracking=True, pretrain=False):
        super(BaselineTrain, self).__init__()
        self.jigsaw = jigsaw
        self.lbda = lbda
        self.rotation = rotation
        self.tracking = tracking
        print('tracking in baseline train:',tracking)
        self.pretrain = pretrain
        print("USE pre-trained model:",pretrain)

        if isinstance(model_func,str):
            if model_func == 'resnet18':
                self.feature = ResidualNet('ImageNet', 18, 1000, None, tracking=self.tracking)
                self.feature.final_feat_dim = 512
            elif model_func == 'resnet18_pytorch':
                self.feature = resnet18(pretrained=self.pretrain, tracking=self.tracking)
                self.feature.final_feat_dim = 512
            elif model_func == 'resnet50_pytorch':
                self.feature = resnet50(pretrained=self.pretrain, tracking=self.tracking)
                self.feature.final_feat_dim = 2048
        else:
            self.feature = model_func()

        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist': #Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)
        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.global_count = 0

        if self.jigsaw:
            self.fc6 = nn.Sequential()
            self.fc6.add_module('fc6_s1',nn.Linear(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            self.fc7 = nn.Sequential()
            self.fc7.add_module('fc7',nn.Linear(9*512,4096))#for resnet
            self.fc7.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_jigsaw = nn.Sequential()
            self.classifier_jigsaw.add_module('fc8',nn.Linear(4096, 35))

        if self.rotation:
            self.fc6 = nn.Sequential()
            self.fc6.add_module('fc6_s1',nn.Linear(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            self.fc7 = nn.Sequential()
            self.fc7.add_module('fc7',nn.Linear(512,128))#for resnet
            self.fc7.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_rotation = nn.Sequential()
            self.classifier_rotation.add_module('fc8',nn.Linear(128, 4))

    def forward(self,x):
        x    = Variable(x.cuda())
        out  = self.feature(x)
        scores  = self.classifier(out.view(x.size(0), -1))
        return scores

    def forward_loss(self, x=None, y=None, patches=None, patches_label=None, unlabel_only=False, label_only=False):
        # import ipdb; ipdb.set_trace()
        if not unlabel_only:
            scores = self.forward(x)
            y = Variable(y.cuda())
            pred = torch.argmax(scores, dim=1)

            if torch.cuda.is_available():
                acc = (pred == y).type(torch.cuda.FloatTensor).mean().item()
            else:
                acc = (pred == y).type(torch.FloatTensor).mean().item()

        if label_only:
            return self.loss_fn(scores, y), acc

        if self.jigsaw:
            B,T,C,H,W = patches.size()#torch.Size([16, 9, 3, 64, 64])
            patches = patches.view(B*T,C,H,W).cuda()#torch.Size([144, 3, 64, 64])
            patch_feat = self.feature(patches)#torch.Size([144, 512, 1, 1])

            x_ = patch_feat.view(B,T,-1)#torch.Size([16, 9, 512])
            x_ = x_.transpose(0,1)#torch.Size([9, 16, 512])

            x_list = []
            for i in range(9):
                z = self.fc6(x_[i])#torch.Size([16, 512])
                z = z.view([B,1,-1])#torch.Size([16, 1, 512])
                x_list.append(z)

            x_ = torch.cat(x_list,1)#torch.Size([16, 9, 512])
            x_ = self.fc7(x_.view(B,-1))#torch.Size([16, 9*512])
            x_ = self.classifier_jigsaw(x_)

            y_ = patches_label.view(-1).cuda()

            pred = torch.max(x_,1)
            acc_jigsaw = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)
            if unlabel_only:
                return self.loss_fn(x_,y_), acc_jigsaw
            else:
                return self.loss_fn(scores, y), self.loss_fn(x_,y_), acc, acc_jigsaw
        elif self.rotation:
            B,R,C,H,W = patches.size()#torch.Size([16, 4, 3, 224, 224])
            patches = patches.view(B*R,C,H,W).cuda()
            x_ = self.feature(patches)#torch.Size([64, 512, 1, 1])
            x_ = x_.squeeze()
            x_ = self.fc6(x_)
            x_ = self.fc7(x_)#64,128
            x_ = self.classifier_rotation(x_)#64,4
            pred = torch.max(x_,1)
            y_ = patches_label.view(-1).cuda()
            acc_jigsaw = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)
            if unlabel_only:
                return self.loss_fn(x_,y_), acc_jigsaw
            else:
                return self.loss_fn(scores, y), self.loss_fn(x_,y_), acc, acc_jigsaw
        else:
            return self.loss_fn(scores, y), acc
    
    def train_loop(self, epoch, train_loader, optimizer, writer, scheduler=None, base_loader_u=None):
        print_freq = min(50,len(train_loader))
        avg_loss=0
        avg_loss_proto=0
        avg_loss_jigsaw=0
        avg_loss_rotation=0
        avg_acc_proto=0
        avg_acc_jigsaw=0
        avg_acc_rotation=0

        if base_loader_u is not None:
            for i,inputs in enumerate(zip(train_loader,base_loader_u)):
                self.global_count += 1
                x = inputs[0][0]
                y = inputs[0][1]
                optimizer.zero_grad()
                loss_proto, acc = self.forward_loss(x, y, label_only=True)

                if self.jigsaw:
                    loss_jigsaw, acc_jigsaw = self.forward_loss(patches=inputs[1][2], patches_label=inputs[1][3], unlabel_only=True)
                    loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_jigsaw
                    writer.add_scalar('train/loss_proto', float(loss_proto.data.item()), self.global_count)
                    writer.add_scalar('train/loss_jigsaw', float(loss_jigsaw), self.global_count)
                elif self.rotation:
                    loss_rotation, acc_rotation = self.forward_loss(patches=inputs[1][2], patches_label=inputs[1][3], unlabel_only=True)
                    loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_rotation
                    writer.add_scalar('train/loss_proto', float(loss_proto.data.item()), self.global_count)
                    writer.add_scalar('train/loss_rotation', float(loss_rotation), self.global_count)
                else:
                    loss, acc = self.forward_loss(x,y)
                writer.add_scalar('train/loss', float(loss.data.item()), self.global_count)
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], self.global_count)

                avg_loss = avg_loss+loss.data#[0]
                avg_acc_proto = avg_acc_proto+acc

                writer.add_scalar('train/acc_cls', acc, self.global_count)
                if self.jigsaw:
                    avg_loss_proto += loss_proto.data
                    avg_loss_jigsaw += loss_jigsaw
                    avg_acc_jigsaw = avg_acc_jigsaw+acc_jigsaw
                    writer.add_scalar('train/acc_jigsaw', acc_jigsaw, self.global_count)
                elif self.rotation:
                    avg_loss_proto += loss_proto.data
                    avg_loss_rotation += loss_rotation
                    avg_acc_rotation = avg_acc_rotation+acc_rotation
                    writer.add_scalar('train/acc_rotation', acc_rotation, self.global_count)

                if (i+1) % print_freq==0:
                    if self.jigsaw:
                        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Cls {:f} | Loss Jigsaw {:f} | Acc Cls {:f} | Acc Jigsaw {:f}'.\
                            format(epoch+1, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_proto/float(i+1), \
                                    avg_loss_jigsaw/float(i+1), avg_acc_proto/float(i+1), avg_acc_jigsaw/float(i+1)))
                    elif self.rotation:
                        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Cls {:f} | Loss Rotation {:f} | Acc Cls {:f} | Acc Rotation {:f}'.\
                            format(epoch+1, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_proto/float(i+1), \
                                    avg_loss_rotation/float(i+1), avg_acc_proto/float(i+1), avg_acc_rotation/float(i+1)))
                    else:
                        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Acc Cls {:f}'.format(epoch+1, i+1, \
                                        len(train_loader), avg_loss/float(i+1), avg_acc_proto/float(i+1)  ))

        else:
            for i, inputs in enumerate(train_loader):
                self.global_count += 1
                x = inputs[0]
                y = inputs[1]
                optimizer.zero_grad()
                if self.jigsaw:
                    loss_proto, loss_jigsaw, acc, acc_jigsaw = self.forward_loss(x, y, inputs[2], inputs[3])
                    loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_jigsaw
                    writer.add_scalar('train/loss_proto', float(loss_proto.data.item()), self.global_count)
                    writer.add_scalar('train/loss_jigsaw', float(loss_jigsaw), self.global_count)
                elif self.rotation:
                    loss_proto, loss_rotation, acc, acc_rotation = self.forward_loss(x, y, inputs[2], inputs[3])
                    loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_rotation
                    writer.add_scalar('train/loss_proto', float(loss_proto.data.item()), self.global_count)
                    writer.add_scalar('train/loss_rotation', float(loss_rotation), self.global_count)
                else:
                    loss, acc = self.forward_loss(x,y)
                writer.add_scalar('train/loss', float(loss.data.item()), self.global_count)
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], self.global_count)

                avg_loss = avg_loss+loss.data
                avg_acc_proto = avg_acc_proto+acc

                writer.add_scalar('train/acc_cls', acc, self.global_count)
                if self.jigsaw:
                    avg_loss_proto += loss_proto.data
                    avg_loss_jigsaw += loss_jigsaw
                    avg_acc_jigsaw = avg_acc_jigsaw+acc_jigsaw
                    writer.add_scalar('train/acc_jigsaw', acc_jigsaw, self.global_count)
                elif self.rotation:
                    avg_loss_proto += loss_proto.data
                    avg_loss_rotation += loss_rotation
                    avg_acc_rotation = avg_acc_rotation+acc_rotation
                    writer.add_scalar('train/acc_rotation', acc_rotation, self.global_count)

                if (i+1) % print_freq==0:
                    if self.jigsaw:
                        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Cls {:f} | Loss Jigsaw {:f} | Acc Cls {:f} | Acc Jigsaw {:f}'.\
                            format(epoch+1, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_proto/float(i+1), \
                                    avg_loss_jigsaw/float(i+1), avg_acc_proto/float(i+1), avg_acc_jigsaw/float(i+1)))
                    elif self.rotation:
                        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Cls {:f} | Loss Rotation {:f} | Acc Cls {:f} | Acc Rotation {:f}'.\
                            format(epoch+1, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_proto/float(i+1), \
                                    avg_loss_rotation/float(i+1), avg_acc_proto/float(i+1), avg_acc_rotation/float(i+1)))
                    else:
                        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Acc Cls {:f}'.format(epoch+1, i+1, \
                                        len(train_loader), avg_loss/float(i+1), avg_acc_proto/float(i+1)  ))
                         
    def test_loop(self, val_loader=None):
        if val_loader is not None:
            num_correct = 0
            num_total = 0
            num_correct_jigsaw = 0
            num_total_jigsaw = 0
            for i, inputs in enumerate(val_loader):
                x = inputs[0]
                y = inputs[1]
                if self.jigsaw:
                    loss_proto, loss_jigsaw, acc, acc_jigsaw = self.forward_loss(x, y, inputs[2], inputs[3])
                    loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_jigsaw
                    num_correct_jigsaw = int(acc_jigsaw*len(inputs[3]))
                    num_total_jigsaw += len(inputs[3].view(-1))
                elif self.rotation:
                    loss_proto, loss_rotation, acc, acc_rotation = self.forward_loss(x, y, inputs[2], inputs[3])
                    loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_rotation
                    num_correct_jigsaw = int(acc_jigsaw*len(inputs[3]))
                    num_total_jigsaw += len(inputs[3].view(-1))
                else:
                    loss, acc = self.forward_loss(x,y)
                num_correct += int(acc*x.shape[0])
                num_total += len(y)
            
            if self.jigsaw or self.rotation:
                return num_correct*100.0/num_total, num_correct_jigsaw*100.0/num_total_jigsaw
            else:
                return num_correct*100.0/num_total

        else:
            if self.jigsaw:
                return -1, -1
            elif self.rotation:
                return -1, -1
            else:
                return -1 #no validation, just save model during iteration


# Reference : https://github.com/karfly/learnable-triangulation-pytorch

import os
import argparse
import json

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from network.algebraic import AlgebraicTriangulation
from network.volumetric import VolumetricTriangulation
from utils import MSEsoft, MAELoss, MPJPE

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parser_args():
    parser = argparse.ArgumentParser(description='Learnable Triangulation of Human Pose')

    parser.add_argument('--model_type',  default='alg', type=str,   help='alg: algebraic model, vol: volumetric model')
    parser.add_argument('--num_epochs',  default=9999,  type=int)
    parser.add_argument('--num_steps',   default=1000,  type=int,   help='')
    parser.add_argument('--batch_size',  default=8,     type=int)
    parser.add_argument('--lr_rate',     default=1e-05, type=float, help='learning rate')
    parser.add_argument('--num_workers', default=0,     type=int)
    parser.add_argument('--ckpt_path',   default='',    type=str,   help='checkpoint file saving')
    parser.add_argument('--save_path',   default='',    type=str,   help='checkpoint file saving')
    parser.add_argument('--load_model',  default=False, type=bool)

    parser.add_argument('--num_joints',  default=17,    type=int)
    parser.add_argument('--num_cameras', default=4,     type=int)
    parser.add_argument('--agg_type',    default='sum', type=str,   help='agg_method list: sum, conf, softmax')

    args = parser.parse_args()

    return args


def training(args, model, criterion, optimizer, train_loader, epoch):

    model.train()
    print('============== {} start training =============='.format(epoch))

    epoch_loss = 0.0
    for i, batch in enumerate(train_loader):
        # load data
        batch_images, batch_keypoints_3d_gt, batch_proj_matrices, batch_pelvis = batch
        batch_images          = batch_images.to(device)
        batch_keypoints_3d_gt = batch_keypoints_3d_gt.to(device)
        batch_proj_matrices   = batch_proj_matrices.to(device)

        # run model
        if args.model_type == 'alg':
            batch_keypoints_3d_pred = model(batch_images, batch_proj_matrices)
        elif args.model_type == 'vol':
            batch_pelvis = batch_pelvis.to(device)
            batch_keypoints_3d_pred = model(batch_images, batch_proj_matrices, batch_pelvis)
        
        # loss
        loss = criterion(batch_keypoints_3d_pred, batch_keypoints_3d_gt)
        epoch_loss += loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % args.num_steps == 0:
            print("Epoch {}/{} -- Step {}/{} -- Loss: {:.4f}"
                    .format(epoch+1, args.num_epochs, i+1, len(train_loader), loss.item()))
    
    print("Epoch {}/{} ----  Total Loss : {:.4f}"
            .format(epoch+1, args.num_epochs, epoch_loss/len(train_loader)))



def validation(args, model, criterion, optimizer, valid_loader, epoch):   

    model.eval()
    print('============== {} start validation =============='.format(epoch))
    prediction  = []
    groundtruth = []
    with torch.no_grad():
        epoch_loss = 0.0
        for i, batch in enumerate(valid_loader):
            # load data
            batch_images, batch_keypoints_3d_gt, batch_proj_matrices, batch_pelvis = batch
            batch_images          = batch_images.to(device)
            batch_keypoints_3d_gt = batch_keypoints_3d_gt.to(device)
            batch_proj_matrices   = batch_proj_matrices.to(device)

            # run model
            if args.model_type == 'alg':
                batch_keypoints_3d_pred = model(batch_images, batch_proj_matrices)
            elif args.model_type == 'vol':
                batch_pelvis = batch_pelvis.to(device)
                batch_keypoints_3d_pred = model(batch_images, batch_proj_matrices, batch_pelvis)
            
            prediction.append(batch_keypoints_3d_pred.detach().cpu().numpy())
            groundtruth.append(batch_keypoints_3d_gt.detach().cpu().numpy())

            # loss
            loss = criterion(batch_keypoints_3d_pred, batch_keypoints_3d_gt)
            epoch_loss += loss

            if (i+1) % args.num_steps == 0:
                print("Epoch {}/{} -- Step {}/{} -- Loss: {:.4f}"
                        .format(epoch+1, args.num_epochs, i+1, len(valid_loader), loss.item()))
        
        # evaluation ################### TODO
        #MPJPE(prediction, groundtruth)

        print("Epoch {}/{} ----  Total Loss : {:.4f}"
            .format(epoch+1, args.num_epochs, epoch_loss/len(valid_loader)))



def main(args):

    # model setting
    if args.model_type == 'alg':
        model = AlgebraicTriangulation(args)
        model.cuda()
    elif args.model_type == 'vol':
        model = VolumetricTriangulation(args)
        model.cuda()

    if args.load_model:
        state_dict = torch.load(args.ckpt_path)
        model.load_state_dict(state_dict, strict=True)
        print("Loaded model")

    # criterion
    if args.model_type == 'alg':
        criterion = MSEsoft()
    elif args.model_type == 'vol':
        criterion = MAELoss

    # optimizer
    if args.model_type == 'vol':
        optimizer = optim.Adam([{'params': model.backbone.parameters(),   'lr': 1e-4},
                                {'params': model.singleConv.parameters(), 'lr': 1e-3},
                                {'params': model.volumeConv.parameters(), 'lr': 1e-3}], lr=args.lr_rate)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),   lr=args.lr_rate)

    
    # dataset
    #train_dataset = ################### TODO
    #valid_dataset = ################### TODO

    # data loader
    train_loader = Dataloader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              #collate_fn=################### TODO,
                              num_workers=args.num_workers,
                              pin_memory=True)

    valid_loader = Dataloader(valid_dataset,
                              vatch_size=args.batch_size,
                              shuffle=False,
                              #collate_fn=################### TODO,
                              num_workers=args.num_workers,
                              pin_memory=True)


    # train loop
    for epoch in range(args.num_epochs):
        training(args, model, criterion, optimizer, train_loader, epoch)
        validation(args, model, criterion, optimizer, valid_loader, epoch)

        # saving
        if (epoch+1) % 2 == 0:
            save_path = args.save_path+"{:04}".format(epoch)
            if not os.path.exist(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, "chekcpoint.pth"))

    # save arguments
    with open('arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)



if __name__ == '__main__':
    args = parser_args()
    main(args)
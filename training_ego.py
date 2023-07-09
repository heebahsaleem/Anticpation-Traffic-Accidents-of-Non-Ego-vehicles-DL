import os
import numpy as np
import time

import torch
from torch import optim
from torch.utils import data
from torchsummaryX import summary

from .loss import train_loss_cal, val_loss_cal
from .models import AP

from .utils import dataloaderHidden

from tensorboardX import SummaryWriter
import pandas as pd


def train(args):
    torch.cuda.set_device(args.device)
    args.dec_hidden_size = args.box_enc_size + args.dm_enc_size

    print(">> Setting the Accident Prediction model ... ")
    AP_model = AP(args).to(args.device)
    all_params = AP_model.parameters()
    optimizer = optim.Adam(all_params, lr=args.lr)

    dataloader_params ={
            "batch_size": args.batch_size,
            "shuffle": args.shuffle
        }

    val_set = dataloaderHidden(args, args.test_root)
    print(">> Number of validation samples:", val_set.__len__())
    val_gen = data.DataLoader(val_set, **dataloader_params)

    print(">> Check the Model's architecture")
    summary(AP_model,
            # torch.zeros(1, args.segment_len, args.pred_timesteps, args.dec_hidden_size).to(device)
            torch.zeros(1, args.segment_len, args.pred_timesteps,  args.dec_hidden_size).to(args.device)
            )

    # MODEL TRAINING
    best_val_ap = None
    best_val_hm = None
    before_ATTC = 0.

    inform = np.zeros((args.train_epoch, 6))

    checkpoint_folder = os.path.join(args.checkpoint, args.trial_name)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    bap_model_name = 'best_AP_model_AP.pt'
    bap_full = os.path.join(checkpoint_folder, bap_model_name)

    #bhm_model_name = 'best_AP_model_HM.pt'
    bhm_model_name = args.best_harmonic_model
    bhm_full = os.path.join(checkpoint_folder, bhm_model_name)

    summary_folder = os.path.join(args.summary, args.trial_name)
    if not os.path.exists(summary_folder):
        os.makedirs(summary_folder)
    print(">> Train data root:", os.path.join(args.data_root, args.train_root))
    writer = SummaryWriter(summary_folder)

    for epoch in range(1, args.train_epoch+1):
        print("\n")
        print("=====================================")
        print("// Epoch :", epoch)
        # regenerate the training dataset
        train_set = dataloaderHidden(args, args.train_root)
        train_gen = data.DataLoader(train_set, **dataloader_params)
        print(" Number of training samples:", train_set.__len__())

        start = time.time()

        #===== train
        train_loss, train_AP, train_ATTC = train_loss_cal(epoch, AP_model, optimizer, train_gen, before_ATTC, verbose=True)
        train_hm = 2.0 * train_AP * train_ATTC / (train_AP * train_ATTC)
        writer.add_scalar('data/train_loss', train_loss, epoch)
        writer.add_scalar('data/train_AP', train_AP, epoch)
        writer.add_scalar('data/train_ATTC', train_ATTC, epoch)
        writer.add_scalar('data/train_Harmonic', train_hm, epoch)
        inform[epoch-1,0] = train_AP
        inform[epoch-1,1] = train_ATTC
        inform[epoch-1,2] = train_hm

        #===== validation
        val_loss, val_AP, val_ATTC = val_loss_cal(epoch, AP_model, val_gen, before_ATTC, verbose=True)
        val_hm = 2.0 * val_AP * val_ATTC / (val_AP + val_ATTC)
        writer.add_scalar('data/val_loss', val_loss, epoch)
        writer.add_scalar('data/val_AP', val_AP, epoch)
        writer.add_scalar('data/val_ATTC', val_ATTC, epoch)
        writer.add_scalar('data/val_Harmonic', val_hm, epoch)
        inform[epoch-1,3] = val_AP
        inform[epoch-1,4] = val_ATTC
        inform[epoch-1,5] = val_hm

        before_ATTC = train_ATTC


        # print time
        elipse = time.time() - start
        print("Elipse: ", elipse)


        if best_val_ap is None:
            best_val_ap = val_AP
            torch.save(AP_model.state_dict(), bap_full)
        elif best_val_ap < val_AP:
            best_val_ap = val_AP
            torch.save(AP_model.state_dict(), bap_full)

        if best_val_hm is None:
            best_val_hm = val_hm
            torch.save(AP_model.state_dict(), bhm_full)
        elif best_val_hm < val_hm:
            best_val_hm = val_hm
            torch.save(AP_model.state_dict(), bhm_full)

        # save checkpoint per epoch
        save_name = 'epoch_{:03d}'.format(epoch) + '.pt'
        full = os.path.join(checkpoint_folder, save_name)
        torch.save(AP_model.state_dict(), full)

    df = pd.DataFrame(inform)
    df.to_csv(os.path.join(checkpoint_folder, 'train_val_inform.csv'), index=False)


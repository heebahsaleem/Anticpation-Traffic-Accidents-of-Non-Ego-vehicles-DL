import os
import numpy as np
import time
from tqdm import tqdm
from glob import glob

import torch
from torch.utils import data

from .models import FOL
from lib.utils import dataloader

import pickle

def get_fol_output(args, fol_model, dataset_gen, save_path):
    fol_model.eval()

    loader = tqdm(dataset_gen, total=len(dataset_gen))

    n = 0
    times = []
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):
            n += 1
            result = {}

            input_bbox, input_dm, input_ego_motion, target_risk_score, file_name = data

            batch_size = input_bbox.size()[0]
            length = input_bbox.size()[1]

            if args.enc_concat_type == 'cat':
                args.dec_hidden_size = args.box_enc_size + args.dm_enc_size
            else:
                if args.box_enc_size != args.dm_enc_size:
                    raise ValueError('Box encoder size %d != DM encoder size %d'
                                     % (args.box_enc_size, args.dm_enc_size))
                else:
                    args.dec_hidden_size = args.box_enc_size

            # Initial hidden state : zeros
            box_h = torch.zeros(batch_size, args.box_enc_size).to(args.device)
            dm_h = torch.zeros(batch_size, args.dm_enc_size).to(args.device)

            list_all_dec_h = torch.zeros(length, args.pred_timesteps, args.dec_hidden_size).to(args.device)

            for i in range(length):
                box = input_bbox[:, i, :]  # box : [batch_size, 4]
                dm = input_dm[:, i, :]  # dm : [batch_size, 4]
                if i > 0:
                    pre = time.time()
                predicts, box_h, dm_h, all_dec_h = fol_model.predict(box, dm, box_h, dm_h)
                if i > 0:
                    cur = time.time()
                    times.append(cur - pre)
                list_all_dec_h[i, :, :] = all_dec_h[0, :, :]

            # print("list_all_dec_h:", list_all_dec_h.shape) #if concat [length, predic_timesteps, 1024]
            save_array = list_all_dec_h.cpu().detach().numpy()
            target_risk_score = target_risk_score.cpu().detach().numpy()

            result['hidden_state'] = save_array  # if concat [length, predic_timesteps, 1024]
            result['target_risk_score'] = target_risk_score[0]

            # save_file = save_path + "{:04}".format(n) + ".pkl"
            if len(file_name[0].split('/')) > 2:
                save_file = os.path.join(save_path, os.path.split(file_name[0])[-1])
            else:
                save_file = os.path.join(save_path, file_name[0])

            with open(save_file, 'wb') as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    times = np.array(times)
    print("FPS: {}, STD: {}, Length: {}".format(np.mean(times),
                                                np.std(times),
                                                len(times)))

    return


def extraction(args):
    # initialize model
    fol_model = FOL(args).to(args.device)
    fol_model.load_state_dict(torch.load(args.best_fol_model))

    dataloader_params = {
        "batch_size": args.batch_size,
        "shuffle": args.shuffle,
    }
    save_root = os.path.join(args.data_root, args.hiddens)
    existing = glob(os.path.join(save_root, '**/*.pkl'), recursive=True)
    if args.overwrite == 'FALSE' and len(existing) > 0:
        print('Files already exist in save root (Overwrite FALSE) {}'.format(save_root))
        return 0

    targets = [args.train_root, args.test_root]
    for t in targets:
        dataset = dataloader(args, t)
        save_to = os.path.join(args.data_root, args.hiddens, t)
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        if dataset.__len__() > 0:
            print(">> Number of dataset:", dataset.__len__())
            dataset_gen = data.DataLoader(dataset, **dataloader_params)
            input_bbox, input_dm, input_ego_motion, target_risk_score, file_name = dataset.__getitem__(1)
            print(" -- input_bbox: ", input_bbox.shape)  # [length, 4]
            print(" -- input_dm: ", input_dm.shape)  # [length, 4]
            print(" -- target_risk_score: ", target_risk_score.shape)  # [1]
            get_fol_output(args, fol_model, dataset_gen, save_to)

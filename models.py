import copy
import torch
from torch import nn, optim
from torch.nn import functional as F

'''
    Encoding means to convert data into a required format.
    
    Output:The output of the encoder, the hidden state, is the state of the last RNN timestep.
    
'''
class EncoderGRU(nn.Module):
    def __init__(self, args):
        super(EncoderGRU, self).__init__()
        self.args = args
        self.enc = nn.GRUCell(input_size=self.args.input_embed_size, #Number of features of input.
                              hidden_size=self.args.enc_hidden_size) #Number of features of hidden state.

    def forward(self, embedded_input, h_init):
        '''
        The encoding process
        Params:
            x: input feature, (batch_size, time, feature dims)
            h_init: initial hidden state, (batch_size, enc_hidden_size)
        Returns:
            h: updated hidden state of the next time step, (batch.size, enc_hiddden_size)
        '''
        h = self.enc(embedded_input, h_init)
        return h #new hidden state of the next timestep


class DecoderGRU(nn.Module):

    def __init__(self, args):
        super(DecoderGRU, self).__init__()
        self.args = args
        self.device = args.device
        # PREDICTOR INPUT FC
        self.hidden_to_pred_input = nn.Sequential(nn.Linear(self.args.dec_hidden_size,
                                                            self.args.predictor_input_size),
                                                  nn.ReLU())

        # PREDICTOR DECODER
        self.dec = nn.GRUCell(input_size=self.args.predictor_input_size,
                              hidden_size=self.args.dec_hidden_size)

        # PREDICTOR OUTPUT
        if self.args.non_linear_output:
            self.hidden_to_pred = nn.Sequential(nn.Linear(self.args.dec_hidden_size,
                                                          self.args.pred_dim),
                                                nn.Tanh())
        else:
            self.hidden_to_pred = nn.Linear(self.args.dec_hidden_size,
                                            self.args.pred_dim)

    def forward(self, h, embedded_ego_pred=None):
        '''
        A RNN preditive model for future observation prediction
        Params:
            h: hidden state tensor from the encoder, (batch_size, enc_hidden_size)
            embedded_ego_pred: (batch_size, pred_timesteps, input_embed_size)
        '''
        output = torch.zeros(h.shape[0], self.args.pred_timesteps, self.args.pred_dim).to(self.device)

        all_pred_h = torch.zeros([h.shape[0], self.args.pred_timesteps, self.args.dec_hidden_size]).to(self.device)
        all_pred_inputs = torch.zeros([h.shape[0], self.args.pred_timesteps, self.args.predictor_input_size]).to(self.device)

        # initial predict input is zero???
        pred_inputs = torch.zeros(h.shape[0], self.args.predictor_input_size).to(self.device)  # self.hidden_to_pred_input(h)
        for i in range(self.args.pred_timesteps):
            if self.args.with_ego:
                pred_inputs = (embedded_ego_pred[:, i,
                               :] + pred_inputs) / 2  # average concat of future ego motion and prediction inputs
            all_pred_inputs[:, i, :] = pred_inputs
            h = self.dec(pred_inputs, h)

            pred_inputs = self.hidden_to_pred_input(h)

            all_pred_h[:, i, :] = h

            output[:, i, :] = self.hidden_to_pred(h)

        return output, all_pred_h, all_pred_inputs



class FOL(nn.Module):
    '''Future object localization module'''

    def __init__(self, args):
        super(FOL, self).__init__()

        # get args and process
        self.args = copy.deepcopy(args)
        self.box_enc_args = copy.deepcopy(args)
        self.dm_enc_args = copy.deepcopy(args)
        self.device = args.device
        self.args.dec_hidden_size = self.args.box_enc_size + self.args.dm_enc_size


        self.box_enc_args.enc_hidden_size = self.args.box_enc_size
        self.dm_enc_args.enc_hidden_size = self.args.dm_enc_size

        # initialize modules
        self.box_encoder = EncoderGRU(self.box_enc_args)
        self.dm_encoder = EncoderGRU(self.dm_enc_args)
        self.args.non_linear_output = True
        self.predictor = DecoderGRU(self.args)

        # initialize other layers
        # self.leaky_relu = nn.LeakyReLU(0.1)
        self.box_embed = nn.Sequential(nn.Linear(4, self.args.input_embed_size),  # size of box input is 4
                                       nn.ReLU())  # nn.LeakyReLU(0.1)
        self.dm_embed = nn.Sequential(nn.Linear(4, self.args.input_embed_size),  # size of DM input is 2=1*1*2
                                        nn.ReLU())  # nn.LeakyReLU(0.1)
        self.ego_pred_embed = nn.Sequential(nn.Linear(3, self.args.input_embed_size),  # size of ego input is 3
                                            nn.ReLU())  # nn.LeakyReLU(0.1)

    def forward(self, box, dm):  # ego_motion):
        '''
        The RNN encoder decoder model rewritten from fvl2019icra-keras
        Params:
            box: (batch_size, segment_len, 4)
            DM: (batch_size, segment_len, 1, 1, 2)
            ego_pred: (batch_size, segment_len, pred_timesteps, 3) or None

            for training and validation, segment_len is large, e.g. 10
            for online testing, segment_len=1
        return:
            fol_predictions: predicted with shape (batch_size, segment_len, pred_timesteps, pred_dim)
        '''
        self.args.batch_size = box.shape[0]
        if len(dm.shape) > 3:
            dm = dm.view(self.args.batch_size, self.args.segment_len, -1)
        embedded_box_input = self.box_embed(box)
        embedded_dm_input = self.dm_embed(dm)

        # initialize hidden states as zeros
        box_h = torch.zeros(self.args.batch_size, self.args.box_enc_size).to(self.device)
        dm_h = torch.zeros(self.args.batch_size, self.args.dm_enc_size).to(self.device)

        # a zero tensor used to save fol prediction
        fol_predictions = torch.zeros(self.args.batch_size,
                                      self.args.segment_len,
                                      self.args.pred_timesteps,
                                      self.args.pred_dim).to(self.device)
        # a zero tensor used to save decoder's hidden state during segment_length
        segment_dec_h = torch.zeros(self.args.batch_size,
                                    self.args.segment_len,
                                    self.args.pred_timesteps,
                                    self.args.dec_hidden_size)

        # run model iteratively, predict T future frames at each time
        for i in range(self.args.segment_len):
            # Box and Differential Motion Encode
            box_h = self.box_encoder(embedded_box_input[:, i, :], box_h)
            dm_h = self.dm_encoder(embedded_dm_input[:, i, :], dm_h)

            # Concat
            hidden_state = torch.cat((box_h, dm_h), dim=1)


            # Decode

            output, all_dec_h, _ = self.predictor(hidden_state, None)

            fol_predictions[:, i, :, :] = output
            segment_dec_h[:, i, :, :] = all_dec_h

        return fol_predictions, segment_dec_h

    def predict(self, box, dm, box_h, dm_h):
        '''
        predictor function, run forward inference to predict the future bboxes
        Params:
            box: (batch_size, 4)
            dm: (batch_size, 4)
            ego_pred: (batch_size, pred_timesteps, 3)
        return:
            box_changes:()
            box_h,
            dm_h
        '''
        # self.args.batch_size = box.shape[0]
        if len(dm.shape) > 3:
            dm = dm.view(self.args.batch_size, -1)
        embedded_box_input = self.box_embed(box)
        embedded_dm_input = self.dm_embed(dm)
        embedded_ego_input = None

        # run model iteratively, predict 5 future frames at each time
        box_h = self.box_encoder(embedded_box_input, box_h)
        dm_h = self.dm_encoder(embedded_dm_input, dm_h)

        hidden_state = torch.cat((box_h, dm_h), dim=1)


        box_changes, all_dec_h, _ = self.predictor(hidden_state, embedded_ego_input)

        # all_dec_h = [batch_size, pred_timesteps, hidden_state_size]
        return box_changes, box_h, dm_h, all_dec_h


class FOL_single_input(nn.Module):
    '''Future object localization module'''

    def __init__(self, args):
        super(FOL_single_input, self).__init__()

        # get args and process
        self.args = copy.deepcopy(args)
        self.box_enc_args = copy.deepcopy(args)
        self.dm_enc_args = copy.deepcopy(args)
        self.device = args.device
        self.args.dec_hidden_size = self.args.box_enc_size + self.args.dm_enc_size


        self.box_enc_args.enc_hidden_size = self.args.box_enc_size
        self.dm_enc_args.enc_hidden_size = self.args.dm_enc_size

        # initialize modules
        self.box_encoder = EncoderGRU(self.box_enc_args)
        self.dm_encoder = EncoderGRU(self.dm_enc_args)
        self.args.non_linear_output = True
        self.predictor = DecoderGRU(self.args)

        # initialize other layers
        # self.leaky_relu = nn.LeakyReLU(0.1)
        self.box_embed = nn.Sequential(nn.Linear(4, self.args.input_embed_size),  # size of box input is 4
                                       nn.ReLU())  # nn.LeakyReLU(0.1)
        self.dm_embed = nn.Sequential(nn.Linear(4, self.args.input_embed_size),  # size of DM input is 2=1*1*2
                                        nn.ReLU())  # nn.LeakyReLU(0.1)
        self.ego_pred_embed = nn.Sequential(nn.Linear(3, self.args.input_embed_size),  # size of ego input is 3
                                            nn.ReLU())  # nn.LeakyReLU(0.1)

    def forward(self, inputs):  # ego_motion):
        '''
        The RNN encoder decoder model rewritten from fvl2019icra-keras
        Params:
            box: (batch_size, segment_len, 4)
            DM: (batch_size, segment_len, 1, 1, 2)
            ego_pred: (batch_size, segment_len, pred_timesteps, 3) or None

            for training and validation, segment_len is large, e.g. 10
            for online testing, segment_len=1
        return:
            fol_predictions: predicted with shape (batch_size, segment_len, pred_timesteps, pred_dim)
        '''
        box = inputs[:,:4]
        dm = inputs[:,4:8]
        self.args.batch_size = box.shape[0]
        if len(dm.shape) > 3:
            dm = dm.view(self.args.batch_size, self.args.segment_len, -1)
        embedded_box_input = self.box_embed(box)
        embedded_dm_input = self.dm_embed(dm)

        # initialize hidden states as zeros
        box_h = torch.zeros(self.args.batch_size, self.args.box_enc_size).to(self.device)
        dm_h = torch.zeros(self.args.batch_size, self.args.dm_enc_size).to(self.device)

        # a zero tensor used to save fol prediction
        fol_predictions = torch.zeros(self.args.batch_size,
                                      self.args.segment_len,
                                      self.args.pred_timesteps,
                                      self.args.pred_dim).to(self.device)
        # a zero tensor used to save decoder's hidden state during segment_length
        segment_dec_h = torch.zeros(self.args.batch_size,
                                    self.args.segment_len,
                                    self.args.pred_timesteps,
                                    self.args.dec_hidden_size)

        # run model iteratively, predict T future frames at each time
        for i in range(self.args.segment_len):
            # Box and Differential Motion Encode
            box_h = self.box_encoder(embedded_box_input[:, i, :], box_h)
            dm_h = self.dm_encoder(embedded_dm_input[:, i, :], dm_h)

            # Concat
            hidden_state = torch.cat((box_h, dm_h), dim=1)


            # Decode

            output, all_dec_h, _ = self.predictor(hidden_state, None)

            fol_predictions[:, i, :, :] = output
            segment_dec_h[:, i, :, :] = all_dec_h

        return fol_predictions, segment_dec_h

    def predict(self, inputs):
        '''
        predictor function, run forward inference to predict the future bboxes
        Params:
            box: (batch_size, 4)
            dm: (batch_size, 4)
            ego_pred: (batch_size, pred_timesteps, 3)
        return:
            box_changes:()
            box_h,
            dm_h
        '''
        # inputs (batch_size, 4 + 4 + len_box_hidden + len_dm_hidden)
        # len_hidden 512
        # 4 + 4 + 512 + 512 = 1032
        box = inputs[:,:4]
        dm = inputs[:,4:8]
        box_h = inputs[:,8:520]
        dm_h = inputs[:,520:]
        
        # self.args.batch_size = box.shape[0]
        if len(dm.shape) > 3:
            dm = dm.view(self.args.batch_size, -1)
        embedded_box_input = self.box_embed(box)
        embedded_dm_input = self.dm_embed(dm)
        embedded_ego_input = None

        # run model iteratively, predict 5 future frames at each time
        box_h = self.box_encoder(embedded_box_input, box_h)
        dm_h = self.dm_encoder(embedded_dm_input, dm_h)

        hidden_state = torch.cat((box_h, dm_h), dim=1)


        box_changes, all_dec_h, _ = self.predictor(hidden_state, embedded_ego_input)

        # all_dec_h = [batch_size, pred_timesteps, hidden_state_size]
        return box_changes, box_h, dm_h, all_dec_h


class AP(nn.Module):
    # Ego-Involved Accident Anticipation
    def __init__(self, args):      #initialization of layers and decoder hidden size
        super(AP, self).__init__()

        self.args = copy.deepcopy(args) #a new copy of all objects inside
        self.device = args.device
        self.avgpool = nn.AvgPool1d(self.args.pred_timesteps)


        self.args.dec_hidden_size = self.args.box_enc_size + self.args.dm_enc_size #512+512

        # ------------- layer 1
        # self.feature_to_output = nn.Sequential(nn.Linear(self.args.dec_hidden_size, 1), nn.Sigmoid())

        # ------------- layer 2
        self.fc_layer = nn.Linear(self.args.dec_hidden_size, self.args.future_context_size)
        self.feature_to_output = nn.Sequential(nn.Linear(self.args.future_context_size, 1), nn.Sigmoid())

    def forward(self, input_all_dec_h):
        '''
            input_all_dec_h : [batch_size, max_length, pred_timesteps, 1024]
        '''
        length = input_all_dec_h.size()[1]  # object's detection period
        batch_size = input_all_dec_h.size()[0]
        risk_scores = torch.zeros(batch_size, length, 1).to(self.device)
        for i in range(length):
            all_dec_h = input_all_dec_h[:, i, :,
                        :]  # batch(1) * frame(n) * timestep(10) * feature(1024) (feature = hidden_state_size)

            all_dec_h = all_dec_h.permute(0, 2, 1)  # batch(1) * feature(1024) * timestep(10)
            pool_dec_h = self.avgpool(all_dec_h)  # batch(1) * feature(1024) * timestep(1)

            pool_dec_h = pool_dec_h.permute(0, 2, 1)  # batch(1) * timestep(1) * feature(1024)
            pool_dec_h = pool_dec_h.view([pool_dec_h.size()[0], pool_dec_h.size()[
                -1]])  # [batch_size, hidden_state_size]   (feature = hidden_state_size)

            future_context_feature = self.fc_layer(pool_dec_h)  # [batch_size, future_context_size]
            risk_score = self.feature_to_output(future_context_feature)

            risk_scores[:, i, :] = risk_score

        return risk_scores

    def predict(self, box, dm, box_h, dm_h, fol_model):
        fol_model.eval()  # use pre-trained fol model ( fol model already training !!! )

        box_changes, box_h, dm_h, all_dec_h = fol_model.predict(box, dm, box_h, dm_h)
        ################################
        all_dec_h = all_dec_h.permute(0, 2, 1)

        pool_dec_h = self.avgpool(all_dec_h)
        pool_dec_h = pool_dec_h.permute(0, 2, 1)
        pool_dec_h = pool_dec_h.view([pool_dec_h.size()[0], pool_dec_h.size()[-1]])  # [1, hidden_state_size]
        future_context_feature = self.fc_layer(pool_dec_h)  # [batch_size, future_context_size]
        risk_score = self.feature_to_output(future_context_feature)
        return risk_score, box_h, dm_h



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process


import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
from tracker.multitracker import JDETracker_management
from models.decode import mot_decode
import copy
from models.utils import decode_tr_res, compute_avgiou, compute_card_score, compute_card_score_v2
from models.losses import SigmoidFocalloss
from torchvision.models.detection import _utils as det_utils


class Mot_Sche_Loss(torch.nn.Module):
    def __init__(self, opt):
        super(Mot_Sche_Loss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        if not self.opt.freeze_detector:
            self.s_det = nn.Parameter(-1.85 * torch.ones(1))
            self.s_id = nn.Parameter(-1.05 * torch.ones(1))

        if opt.add_decision_head:
            self.s_sche = nn.Parameter(-1.05 * torch.ones(1))
        self.crit_sche_dt_to = SigmoidFocalloss(alpha=-1, gamma=0, activation=None, reduction="mean")


    def forward(self, outputs, batch, tr_results, tr_results_to, labels, regression_targets):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        sche_loss, sche_loss_tmp = 0, 0

        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.freeze_detector:
                if not opt.mse_loss:
                    output['hm'] = _sigmoid(output['hm'])

                hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
                if opt.wh_weight > 0:
                    wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks

                if opt.reg_offset and opt.off_weight > 0:
                    off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                              batch['ind'], batch['reg']) / opt.num_stacks

                if opt.id_weight > 0:
                    id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                    id_head = id_head[batch['reg_mask'] > 0].contiguous()
                    id_head = self.emb_scale * F.normalize(id_head)
                    id_target = batch['ids'][batch['reg_mask'] > 0]

                    id_output = self.classifier(id_head).contiguous()
                    id_loss += self.IDLoss(id_output, id_target)

                det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss

            if opt.add_decision_head:
                fr_ind_vec, dt_tlwhs_vec, dt_ids_vec = decode_tr_res(tr_results)
                _, to_tlwhs_vec, to_ids_vec = decode_tr_res(tr_results_to)
                dt_to_iou_scores, dt_to_card_scores = list(), list()

                for ind in range(opt.batch_size):
                    dt_to_iou_scores.append(compute_avgiou(dt_tlwhs_vec[ind], to_tlwhs_vec[ind], thres=0.5))
                    dt_to_card_scores.append(compute_card_score(dt_tlwhs_vec[ind].shape[0], to_tlwhs_vec[ind].shape[0]))

                dt_to_card_scores_tensor = torch.cat(dt_to_card_scores, dim=0).to(output['initial'].device)
                dt_to_iou_scores_tensor = torch.cat(dt_to_iou_scores, dim=0).to(output['initial'].device)

                dt_to_scores = dt_to_card_scores_tensor * dt_to_iou_scores_tensor
                dt_to_scores = torch.ge(dt_to_scores, 0.7).float()

                sche_loss = self.crit_sche_dt_to(output['initial'], 1 - dt_to_scores)

        loss = 0
        loss_stats = dict()
        if opt.add_decision_head:
            loss += sche_loss * 1.0
            loss_stats.update({'decision_loss': sche_loss})
        if not opt.freeze_detector:
            loss = loss.unsqueeze(0)
            detector_loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (
                        self.s_det + self.s_id)
            detector_loss *= 0.5
            loss += detector_loss
            loss_stats.update(
                {'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
            )

        loss_stats.update({'loss': loss})

        return loss, loss_stats

class ModelWithLoss_w_tracking(torch.nn.Module):
    def __init__(self, model, loss, opt):
        super(ModelWithLoss_w_tracking, self).__init__()
        self.model = model
        self.loss = loss
        self.opt = opt
        self.key_fr = torch.zeros((opt.batch_size, 64, opt.output_w, opt.output_h), device=self.opt.device)
        self.trackers = [None]*opt.batch_size
        self.heights, self.widths = [opt.img_size[1]]*opt.batch_size,[opt.img_size[0]]*opt.batch_size
        self.batch_size = opt.batch_size
        self.results = [list()]*opt.batch_size # Detect + Track
        self.results_to = [list()] * opt.batch_size # Track Only
        self.min_box_area = opt.min_box_area
        self.key_selection_flag = False
        self.image_shapes = [tuple()]*opt.batch_size
        self.bboxes = torch.zeros((opt.batch_size, opt.K, 4), device=self.opt.device)
        #FROM FasterRCNN
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            opt.K,
            0.25)
        self.proposal_matcher = det_utils.Matcher(
            0.5,
            0.5,
            allow_low_quality_matches=False)


        bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)


    def change_keyupdate_option(self, flag:bool):
        self.key_selection_flag = True

    def forward(self, batch):
        for ind, k in enumerate(batch['fr_ind']):
            if torch.eq(k,1).detach().item(): # fr_ind == 0000001 (first frame)
                #TODO: initialize Tracklet
                self.trackers[ind] = JDETracker_management(self.opt, frame_rate=batch['frame_rate'][ind].detach().item())
                self.heights[ind] = batch['input_h'][ind].detach().item()
                self.widths[ind] = batch['input_w'][ind].detach().item()
                self.results[ind] = list()
                self.results_to[ind] = list()
                self.image_shapes[ind] = (self.heights[ind], self.widths[ind])
                self.key_fr[ind] = torch.zeros_like(self.key_fr[ind])
                self.bboxes[ind] = torch.zeros_like(self.bboxes[ind], device=self.opt.device)


        outputs = self.model(x=batch['input'], key=self.key_fr, bboxes=self.bboxes, image_shapes=self.image_shapes)

        # Tracking management step1 : get detections & embedding & initial_flag
        with torch.no_grad():
            output = self.model(x=batch['input'], key=self.key_fr, bboxes=self.bboxes, image_shapes=self.image_shapes, act_buffer=False)[-1]
            for ind in range(self.batch_size):
                trackers_temp = copy.deepcopy(self.trackers[ind])

                output_i = dict()
                for key, value in output.items():
                    output_i.update({key:value[ind].unsqueeze(0)})

                # Detect and Track
                online_targets = self.trackers[ind].update(batch['input'][ind].unsqueeze(0), output_i, self.widths[ind],
                                                           self.heights[ind], skip_det=False)
                online_tlwhs = []
                online_ids = []
                # online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                self.results[ind].append((batch['fr_ind'][ind], online_tlwhs, online_ids))

                # Track Only
                online_targets_to = trackers_temp.update(batch['input'][ind].unsqueeze(0), output_i, self.widths[ind],
                                                           self.heights[ind], skip_det=True)
                online_tlwhs_to = []
                online_ids_to = []
                for t in online_targets_to:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                        online_tlwhs_to.append(tlwh)
                        online_ids_to.append(tid)
                self.results_to[ind].append((batch['fr_ind'][ind], online_tlwhs_to, online_ids_to))

                if self.key_selection_flag:
                    flag = output['initial'][ind].detach().item() >= 0.5
                else:
                    val = torch.rand(1).item()
                    flag = (val>= 0.5)

                if flag: # Tracking-by-detection (TD)
                    self.key_fr[ind] = output['dla_feat'][ind].detach() # Key frame update
                else: # Tracking-by-motion (TM)
                    self.trackers[ind] = trackers_temp # Tracker state change

        loss, loss_stats = self.loss(outputs, batch, self.results, self.results_to, None, None)
        return outputs[-1], loss, loss_stats


class Mot_Sche_Trainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss_w_tracking(model, self.loss, opt)
        self.optimizer.add_param_group({'params': self.loss.parameters()})

        self.key_selection_flag = False

    def _get_losses(self, opt):

        loss_states = ['loss']
        if not opt.freeze_detector:
            loss_states.extend(['hm_loss', 'wh_loss', 'off_loss', 'id_loss'])
        if opt.add_decision_head:
            loss_states.append('decision_loss')
        loss = Mot_Sche_Loss(opt)
        return loss_states, loss

    def change_keyupdate_option(self, flag:bool):
        self.model_with_loss.change_keyupdate_option(flag)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]


    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)


    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()

            if self.opt.freeze_detector:
                for n, m in model_with_loss.model.named_modules():
                    import torch.nn as nn
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        if n.find('initial')!=-1:
                            m.train()
                        if n.find('attention')!=-1:
                            m.train()
                        if n.find('encoder') != -1:
                            m.train()
                        if n.find('bbox_ref') != -1:
                            m.train()
                        if n.find('bbox_belief') != -1:
                            m.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            output, loss, loss_stats = model_with_loss(batch)


            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats, batch

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
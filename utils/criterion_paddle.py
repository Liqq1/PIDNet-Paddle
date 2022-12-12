
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from configs import config


class CrossEntropy(nn.Layer):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(score):
            return sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, score)])
        elif len(score) == 1:
            return sb_weights * self._forward(score[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")

        


class OhemCrossEntropy(nn.Layer):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        
        score = paddle.transpose(score, perm=[0,2,3,1])
        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        
        #score-[6, 19, 1024, 1024]
        #target-[6, 1024, 1024]
        #score = paddle.transpose(score, perm=[0,2,3,1])

        logit = score
        label = target
        if len(label.shape) != len(logit.shape):
            label = paddle.unsqueeze(label, 1)
        n, c, h, w = logit.shape
        label = label.reshape((-1,)).astype('int64')
        valid_mask = (label != 255).astype('int64')
        num_valid = valid_mask.sum()
        label = label * valid_mask

        prob = paddle.nn.functional.softmax(logit, axis=1)
        prob = prob.transpose((1, 0, 2, 3)).reshape((c, -1))

        if self.min_kept < num_valid and num_valid > 0:
            prob = prob + (1 - valid_mask)
            label_onehot = paddle.nn.functional.one_hot(label, c)
            label_onehot = label_onehot.transpose((1, 0))
            prob = prob * label_onehot
            prob = paddle.sum(prob, axis=0) 

            threshold = self.thresh
            if self.min_kept > 0:
                index = prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                threshold_index = int(threshold_index)
                if prob[threshold_index] > threshold:
                    threshold = prob[threshold_index]
                kept_mask = (prob < threshold).astype('int64')
                label = label * kept_mask
                valid_mask = valid_mask * kept_mask
        label = label + (1 - valid_mask) * 255
        label = label.reshape((n, 1, h, w))
        valid_mask = valid_mask.reshape((n, 1, h, w)).astype('float32')
        loss = paddle.nn.functional.softmax_with_cross_entropy(
                          logit, label, ignore_index=255, axis=1)
        loss = loss * valid_mask
        avg_loss = paddle.mean(loss) / max(paddle.mean(valid_mask), 1e-6)

        label.stop_gradient = True
        valid_mask.stop_gradient = True
        return avg_loss


    def forward(self, score, target):
        
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * \
                (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(balance_weights, score, functions)
            ])
        
        elif len(score) == 1:
            return sb_weights * self._ohem_forward(score[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")


def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.shape #[6,1,1024,1024]

    #log_p = bd_pre.permute(0,2,3,1).reshape([1, -1])

    log_p = paddle.transpose(bd_pre, perm=[0,2,3,1])
    log_p = log_p.reshape([1, -1])

    target_t = target.reshape([1, -1]).astype("float32")

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = paddle.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num
    

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss


class BondaryLoss(nn.Layer):
    def __init__(self, coeff_bce = 20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce
        
    def forward(self, bd_pre, bd_gt):

        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss
        
        return loss
    
# if __name__ == '__main__':
#     a = torch.zeros(2,64,64)
#     a[:,5,:] = 1
#     pre = torch.randn(2,1,16,16)
    
#     Loss_fc = BondaryLoss()
#     loss = Loss_fc(pre, a.to(torch.uint8))

        
        
        



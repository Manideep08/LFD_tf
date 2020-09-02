import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer

mse = tensorflow.keras.losses.MeanSquaredError(reduction=tensorflow.keras.losses.Reduction.SUM)

class cross_entropy_with_hnm_for_one_class_detection(Model):
    def __init__(self, hnm_ratio, num_output_scales):
        super(cross_entropy_with_hnm_for_one_class_detection, self).__init__()
        self.hnm_ratio = int(hnm_ratio)
        self.num_output_scales = num_output_scales

    def call(self, outputs, targets):

        losses = []
        for i in range(self.num_output_scales):
            pred_score = outputs[i * 2]
            pred_bbox = outputs[i * 2 + 1]
            gt_mask = targets[i * 2]
            gt_label = targets[i * 2 + 1]

            pred_score_softmax = tensorflow.nn.softmax(pred_score, axis=1)
            loss_mask = tensorflow.ones(pred_score_softmax.shape, tensorflow.float32)

            if self.hnm_ratio > 0:
                pos_flag = (gt_label[:, 0, :, :] > 0.5)
                pos_num = tensorflow.math.reduce_sum(tensorflow.cast(pos_flag, dtype=tensorflow.float32)) # get num. of positive examples
            if pos_num > 0:
                neg_flag = (gt_label[:, 1, :, :] > 0.5)
                neg_num = tensorflow.math.reduce_sum(tensorflow.cast(neg_flag, dtype=tensorflow.float32))
                neg_num_selected = min(int(self.hnm_ratio * pos_num), int(neg_num))
                neg_prob = tensorflow.where(neg_flag, pred_score_softmax[:, 1, :, :], \
                tensorflow.zeros_like(pred_score_softmax[:, 1, :, :]))
                neg_prob_sort = tensorflow.sort(tensorflow.reshape(neg_prob, shape=(1, -1)), direction='ASCENDING')
                prob_threshold = neg_prob_sort[0][int(neg_num_selected)]
                neg_grad_flag = (neg_prob <= prob_threshold)
                loss_mask = tensorflow.concat([tensorflow.expand_dims(pos_flag, axis=1), tensorflow.expand_dims(neg_grad_flag, axis=1)], axis=1)
            else:
                neg_choice_ratio = 0.1
                neg_num_selected = int(tensorflow.cast(tensorflow.size(pred_score_softmax[:, 1, :, :]), dtype=tensorflow.float32) * 0.1)
                neg_prob = pred_score_softmax[:, 1, :, :]
                neg_prob_sort = tensorflow.sort(tensorflow.reshape(neg_prob, shape=(1, -1)), direction='ASCENDING')
                prob_threshold = neg_prob_sort[0][int(neg_num_selected)]
                neg_grad_flag = (neg_prob <= prob_threshold)                
                loss_mask = tensorflow.concat([tensorflow.expand_dims(pos_flag, axis=1), tensorflow.expand_dims(neg_grad_flag, axis=1)], axis=1)

            pred_score_softmax_masked = tensorflow.where(loss_mask, pred_score_softmax, tensorflow.zeros_like(pred_score_softmax, dtype=tensorflow.float32))
            pred_score_log = tensorflow.math.log(pred_score_softmax_masked)
            score_cross_entropy = - tensorflow.where(loss_mask, gt_label[:, :2, :, :], tensorflow.zeros_like(gt_label[:, :2, :, :], dtype=tensorflow.float32)) * pred_score_log
            loss_score = tensorflow.math.reduce_sum(score_cross_entropy) / tensorflow.cast(tensorflow.size(score_cross_entropy), tensorflow.float32)

            mask_bbox = gt_mask[:, 2:6, :, :]
            predict_bbox = pred_bbox * mask_bbox
            label_bbox = gt_label[:, 2:6, :, :] * mask_bbox
            # l2 loss of boxes
            # loss_bbox = tensorflow.math.reduce_sum(tensorflow.nn.l2_loss((label_bbox - predict_bbox)) ** 2) / 2
            loss_bbox = mse(label_bbox, predict_bbox) / tensorflow.math.reduce_sum(mask_bbox)

            # Adding only losses relevant to a branch and sending them for back prop
            losses.append(loss_score + loss_bbox)
            # losses.append(loss_bbox)

            

            # Adding all losses and sending to back prop Approach 1
            # loss_cls += loss_score
            # loss_reg += loss_bbox
            # loss_branch.append(loss_score)
            # loss_branch.append(loss_bbox)
            # loss = loss_cls + loss_reg

        return losses

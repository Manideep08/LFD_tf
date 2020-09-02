# -*- coding: utf-8 -*-
import os
import tensorflow
import logging
import time


class Solver(object):
    def __init__(self,
                 task_name,
                 tensorflow_module,
                 trainset_dataiter,
                 scheduler,
                 net,
                 optimizer,
                 gpu_id_list,
                 num_train_loops,
                 loss_criterion,
                 train_metric,
                 display_interval=10,
                 val_evaluation_interval=100,
                 valset_dataiter=None,
                 val_metric=None,
                 num_val_loops=0,
                 pretrained_model_param_path=None,
                 save_prefix=None,
                 start_index=0,
                 model_save_interval=None,
                 train_metric_update_frequency=1):
        self.task_name = task_name
        self.tensorflow_module = tensorflow_module
        self.trainset_dataiter = trainset_dataiter
        self.valset_dataiter = valset_dataiter
        self.scheduler = scheduler
        self.net = net
        self.gpu_id_list = gpu_id_list
        self.optimizer = optimizer
        self.num_train_loops = num_train_loops
        self.num_val_loops = num_val_loops
        self.loss_criterion = loss_criterion
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.display_interval = display_interval
        self.val_evaluation_interval = val_evaluation_interval
        self.save_prefix = save_prefix
        self.start_index = start_index
        self.pretrained_model_param_path = pretrained_model_param_path
        self.model_save_interval = model_save_interval

        self.train_metric_update_frequency = \
            train_metric_update_frequency if train_metric_update_frequency <= \
            display_interval else display_interval

    def fit(self):
        logging.info('Start training in gpu %s.-----------', str(self.gpu_id_list))
        sum_time = 0
        for i in range(self.start_index + 1, self.num_train_loops + 1):
            start = time.time()
            batch = self.trainset_dataiter.next()
            
            # images = tensorflow.Variable(batch.data[0], trainable=False)
            images = batch.data[0]
            images = (images - 127.5) / 127.5

            targets = batch.label

            with tensorflow.GradientTape() as tape:
                outputs = self.net(images)
                loss = self.loss_criterion(outputs, targets)
            
            self.scheduler(i, self.optimizer)
            grads = tape.gradient(loss, self.net.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))

            # self.optimizer.learning_rate(tensorflow.convert_to_tensor(i))
            # self.optimizer.weight_decay(tensorflow.convert_to_tensor(i))

            """the train_metric need to debug"""
            # display training process----------------------------------------
            if i % self.train_metric_update_frequency == 0:
                self.train_metric.update(targets, outputs)

            sum_time += (time.time() - start)

            if i % self.display_interval == 0:

                names, values = self.train_metric.get()

                logging.info('Iter[%d] -- Time elapsed: %.1f s. Speed: %.1f images/s.',
                             i, sum_time, self.display_interval * \
                             self.trainset_dataiter.get_batch_size() / sum_time)
                for name, value in zip(names, values):
                    logging.info('%s: --> %.4f', name, value)

                self.train_metric.reset()
                sum_time = 0

            # evaluate the validation set
            if i % self.val_evaluation_interval == 0 and self.num_val_loops:
                with torch.no_grad():
                    logging.info('Start validating---------------------------')
                    for val_loop in range(self.num_val_loops):
                        val_batch = self.valset_dataiter.next()
                        val_images = val_batch[0].cuda()
                        val_targets = val_batch[1:].cuda()

                        val_outputs = self.net(val_images)

                        self.val_metric.update(val_outputs, val_targets)

                    names, values = self.val_metric.get()
                    logging.info('Iter[%d] validation metric -------------', i)
                    for name, value in zip(names, values):
                        logging.info('%s: --> %.4f', name, value)
                    logging.info('End validating ----------------------------')
                    self.val_metric.reset()

            # save model-----------------------------------------------------
            if i % self.model_save_interval == 0:
                self.net.save(self.save_prefix + '/' + self.task_name + '__' + str(i))
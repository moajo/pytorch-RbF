from typing import Union, List

from torchnet.dataset import TransformDataset

from kernels import GaussianRbFKernel, LaplaceRbFKernel, LinearRbFKernel, CosineRbFKernel, QuadraticRbFKernel, \
    SecantRbFKernel
import numpy as np


class RbFIterator:
    KERNELS = {
        "gau": GaussianRbFKernel(),
        "lap": LaplaceRbFKernel(),
        "lin": LinearRbFKernel(),
        "cos": CosineRbFKernel(),
        "qua": QuadraticRbFKernel(),
        "sec": SecantRbFKernel(),
    }

    def __init__(self, dataset, kernel_type, nu):
        assert kernel_type in self.KERNELS
        self.dataset = dataset
        self.kernel_type = kernel_type
        self.nu = nu

        self.delays = None
        self.current_data_size = None

        # for memorize training result
        self.train_loss = None
        self.train_acc = None
        self.index_buffer = None

        # for memorize validation result
        self.val_loss = None
        self.val_acc = None

        # unused data in current epoch
        self.skip_data_index = None

        self.reset()

    def get_data(self):
        """
        get dataset for current epoch.
        :return: Dataset:-> tuple(data,batch_index)
        """
        use_data_index = (self.delays <= 1).nonzero()[0]
        self.skip_data_index = (self.delays > 1).nonzero()[0]
        self.current_data_size = len(use_data_index)
        self.train_loss = np.zeros(len(self.dataset))
        self.train_acc = np.zeros(len(self.dataset))
        return TransformDataset(use_data_index, lambda i: (self.dataset[i], i))

    def add_train_result(self, batch_index: Union[List[int], np.ndarray], acc: float, loss: float):
        """
        add training result for each batch to buffer.
        :param batch_index: index list of data contained this mini-batch.
        :param acc: average accuracy of this batch.
        :param loss: average loss of this batch.
        :return:
        """
        self.train_loss[batch_index] = loss
        self.train_acc[batch_index] = acc
        self.index_buffer += list(batch_index)

    def add_validation_result(self, acc, loss, batch_size):
        self.val_loss += [loss for _ in range(batch_size)]
        self.val_acc += [acc for _ in range(batch_size)]

    def update_delay_table(self):
        """
        update delay table using train/val buffer. and clear buffers.
        :return:
        """
        nu = self.nu
        d = 1.0
        val_strength = np.mean(self.val_acc)
        x = d / val_strength

        kernel = self.KERNELS[self.kernel_type]
        estimated_tau = kernel.estimate_tau(x, np.array(self.val_acc), np.array(self.val_loss), nu)

        train_index_order = np.array(self.index_buffer)
        delay, idx = kernel.calculate_delay(nu, estimated_tau, val_strength,
                                            self.train_acc[train_index_order],
                                            self.train_loss[train_index_order])
        self.delays[train_index_order[idx]] = delay
        self.delays[self.skip_data_index] -= 1
        self.clean_buffer()

    def reset(self):
        self.delays = np.ones(len(self.dataset))
        self.current_data_size = len(self.dataset)
        self.clean_buffer()

    def clean_buffer(self):
        self.train_loss = np.zeros(len(self.dataset))
        self.train_acc = np.zeros(len(self.dataset))
        self.index_buffer = []
        self.val_loss = []
        self.val_acc = []
        self.skip_data_index = None

import numpy as np


class RbFKernelBase:
    def estimate_tau(self, x, val_acc, val_loss, nu):
        raise NotImplementedError()

    def calculate_delay(self, nu, tau, val_strength, train_acc, train_loss):
        raise NotImplementedError()


class GaussianRbFKernel(RbFKernelBase):
    def estimate_tau(self, x, val_acc, val_loss, nu):
        a_ln = -1 * np.log(val_acc[val_acc >= nu]).sum()
        x_sum_pow = ((val_loss[val_acc >= nu] * val_acc[val_acc >= nu]) ** 2).sum()
        return a_ln / x_sum_pow

    def calculate_delay(self, nu, tau, val_strength, train_acc, train_loss):
        idx = (train_acc >= nu).nonzero()
        nu_gau = np.sqrt(-np.log(nu) / tau)
        res = np.clip(val_strength * nu_gau / train_loss[idx], 1, np.inf)
        return res, idx


class LaplaceRbFKernel(RbFKernelBase):
    def estimate_tau(self, x, val_acc, val_loss, nu):
        a_ln = -1. * np.sum([np.log(a) for a in val_acc if a >= nu])
        x_sum = np.sum([l * x for l, a in zip(val_loss, val_acc) if a >= nu])
        return a_ln / x_sum

    def calculate_delay(self, nu, tau, val_strength, train_acc, train_loss):
        idx = (train_acc >= nu).nonzero()
        nu_lap = np.log(nu)
        res = np.clip(
            -1. * val_strength * nu_lap / (train_loss[idx] * tau)
            , 1, np.inf)
        return res, idx


class LinearRbFKernel(RbFKernelBase):

    def estimate_tau(self, x, val_acc, val_loss, nu):
        a_one = np.sum([(1. - a) for a in val_acc if a >= nu])
        x_sum = np.sum([l * x for l, a in zip(val_loss, val_acc) if a >= nu])
        return a_one / x_sum

    def calculate_delay(self, nu, tau, val_strength, train_acc, train_loss):
        idx = (train_acc >= nu).nonzero()
        nu_lin = (1. - nu)
        res = np.clip(
            val_strength * nu_lin / (train_loss[idx] * tau)
            , 1, np.inf)
        return res, idx


class CosineRbFKernel(RbFKernelBase):

    def estimate_tau(self, x, val_acc, val_loss, nu):
        a_arc = np.sum([np.arccos(2. * a - 1.) for a in val_acc if a >= nu])
        x_sum = np.sum([l * x for l, a in zip(val_loss, val_acc) if a >= nu])
        return a_arc / (np.pi * x_sum)

    def calculate_delay(self, nu, tau, val_strength, train_acc, train_loss):
        idx = (train_acc >= nu).nonzero()
        nu_cos = np.arccos(2 * nu - 1.)
        res = np.clip(
            val_strength * nu_cos / (np.pi * train_loss[idx] * tau)
            , 1, np.inf)
        return res, idx


class QuadraticRbFKernel(RbFKernelBase):

    def estimate_tau(self, x, val_acc, val_loss, nu):
        a_one = np.sum([(1. - a) for a in val_acc if a >= nu])
        x_sum_pow = np.sum([pow(l * x, 2) for l, a in zip(val_loss, val_acc) if a >= nu])
        return a_one / x_sum_pow

    def calculate_delay(self, nu, tau, val_strength, train_acc, train_loss):
        idx = (train_acc >= nu).nonzero()
        nu_qua = np.sqrt((1. - nu) / tau)
        res = np.clip(
            val_strength * nu_qua / train_loss[idx]
            , 1, np.inf)
        return res, idx


class SecantRbFKernel(RbFKernelBase):

    def estimate_tau(self, x, val_acc, val_loss, nu):
        a_sq = np.sum([np.log(1. / a + np.sqrt(1. / a - 1.)) for a in val_acc if a >= nu])
        x_sum = np.sum([l * x for l, a in zip(val_loss, val_acc) if a >= nu])
        return a_sq / x_sum

    def calculate_delay(self, nu, tau, val_strength, train_acc, train_loss):
        idx = (train_acc >= nu).nonzero()
        nu_sec = np.log(1. / nu * (1 + np.sqrt(1 - nu * nu)))
        res = np.clip(
            val_strength * nu_sec / (train_loss[idx] * tau)
            , 1, np.inf)
        return res, idx

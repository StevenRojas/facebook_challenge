import torch
import sys


class HyperHandler:

    """
    Class to define hyper parameters
    @author: Steven Rojas <steven.rojas@gmail.com>
    """

    def __init__(self, config):
        self.config = config

        self.supported_optimizers = {
            "sgd": self.__get_sgd
        }

        self.supported_scheduler = {
            "none": self.__get_none,
            "step_lr": self.__get_step_lr
        }
        # TODO: Support more hyper parameters

    def generate(self, parameters):
        criterion = self.__get_criterion()
        optimizer = self.__get_optimizer(parameters)
        scheduler = self.__get_scheduler(optimizer)
        return criterion, optimizer, scheduler

    def __get_optimizer(self, parameters):
        name = self.config["optimizer"]["name"].lower()
        if name not in self.supported_optimizers.keys():
            sys.exit("[ERROR] Not supported optimizer")
        func = self.supported_optimizers.get(name)
        return func(parameters)

    def __get_scheduler(self, optimizer):
        name = self.config["scheduler"]["name"].lower()
        if name not in self.supported_scheduler.keys():
            sys.exit("[ERROR] Not supported scheduler")
        func = self.supported_scheduler.get(name)
        return func(optimizer)

    def __get_criterion(self):
        if self.config["criterion"].lower() == "cross_entropy":
            return torch.nn.CrossEntropyLoss()

        sys.exit("[ERROR] Not supported criterion")

    def __get_none(self, optional = None):
        return None

    # Optimizers

    def __get_sgd(self, parameters):
        momentum = self.config["optimizer"]["params"]["momentum"]
        weight_decay = self.config["optimizer"]["params"]["weight_decay"]
        lr = self.config["learning_rate"]
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Schedulers

    def __get_step_lr(self, optimizer):
        step_size = self.config["scheduler"]["params"]["step_size"]
        gamma = self.config["scheduler"]["params"]["gamma"]
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

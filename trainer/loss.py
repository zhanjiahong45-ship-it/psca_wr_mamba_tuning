from torch import nn


class CrossEntropy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.crit = nn.CrossEntropyLoss()

    @property
    def ignore_index(self):
        return self.crit.ignore_index

    def forward(self, input, target):
        return self.crit(input.float().reshape(-1, input.size(-1)), target.reshape(-1))
    

class Accuracy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self):
        return "accuracy"

    def forward(self, input, target):
        if input.ndim == target.ndim + 1:
            input = input.argmax(-1)

        return (input == target).mean()

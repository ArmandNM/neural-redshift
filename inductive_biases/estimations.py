from typing import Tuple

import torch

from inductive_biases.data.synthetic import make_grid_points, make_linspace_points


class DecisionSpaceEstimator:
    def __init__(self) -> None:
        self.inputs = None
        self.space_shape = None

    def estimate(self, model: torch.nn.Module, batch_size=1):
        dataloader = torch.utils.data.DataLoader(self.inputs, batch_size=batch_size)

        device = next(model.parameters()).device

        outputs = []
        for data in dataloader:
            data = data.to(device)
            outputs.append(model(data))

        outputs = torch.cat(outputs)
        outputs = outputs.reshape(self.space_shape)
        return outputs


class LinearDecisionSpaceEstimator(DecisionSpaceEstimator):
    def __init__(self, start: Tuple[int], end: Tuple[int], steps: int = 300, integer: bool = False):
        assert len(start) == len(end)
        self.start = start
        self.end = end
        self.steps = steps
        self.inputs = make_linspace_points(start, end, steps)
        if integer:
            self.inputs = self.inputs.to(torch.int64)
        self.space_shape = (steps,) * len(start)


class DiscreteDecisionSpaceEstimator(DecisionSpaceEstimator):
    def __init__(self, start: Tuple[int], end: Tuple[int]):
        assert len(start) == len(end)
        self.start = start
        self.end = end
        self.inputs = make_grid_points(start, end)
        self.space_shape = tuple([end[i] - start[i] + 1 for i in range(len(start))])

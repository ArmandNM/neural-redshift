import torch


class OutputSpaceEstimator:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, steps: int = 300):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.steps = steps
        self.grid_x, self.grid_y = self.make_points_on_grid(x1, y1, x2, y2, steps)

    @staticmethod
    def make_points_on_grid(x1: float, y1: float, x2: float, y2: float, steps: int = 300):
        xs = torch.linspace(x1, x2, steps)
        ys = torch.linspace(y1, y2, steps)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        # TODO (armand.nicolicioiu): Check out torch.cartesian_prod(xs, ys).
        return grid_x, grid_y

    def estimate(self, model: torch.nn.Module, batch_size=1):
        inputs = torch.cat(tuple(torch.dstack([self.grid_x, self.grid_y])))
        dataloader = torch.utils.data.DataLoader(inputs, batch_size=batch_size)

        device = next(model.parameters()).device

        outputs = []
        with torch.no_grad():
            for data in dataloader:
                data = data.to(device)
                outputs.append(model(data).detach())

        outputs = torch.cat(outputs)
        outputs = outputs.detach().reshape(self.grid_x.shape).cpu().numpy()
        return outputs

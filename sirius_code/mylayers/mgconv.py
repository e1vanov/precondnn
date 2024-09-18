import torch
import torch.nn as nn
import torch.nn.functional as F

class MGConv(nn.Module):

    def __init__(self, depth, channels):

        super().__init__()

        assert depth == len(channels)
        assert depth >= 2

        self.depth = depth
        self.channels = channels

        self.refiners = nn.ModuleList([])
        for i in range(1, depth):
            self.refiners.append(nn.Upsample(scale_factor=2.0,
                                             mode='linear'))

        self.coarsers = nn.ModuleList([])
        for i in range(depth - 1):
            self.coarsers.append(nn.Upsample(scale_factor=0.5,
                                             mode='linear'))

        self.convs = nn.ModuleList([])
        self.convs.append(nn.Conv1d(channels[0] + channels[1],
                                    channels[0],
                                    kernel_size=3, padding=1))
        for i in range(1, depth - 1):
            self.convs.append(nn.Conv1d(channels[i - 1] + channels[i] + channels[i + 1],
                                        channels[i],
                                        kernel_size=3, padding=1))
        self.convs.append(nn.Conv1d(channels[depth - 1] + channels[depth - 2],
                                    channels[depth - 1],
                                    kernel_size=3, padding=1))

    def forward(self, x):

        y = [0 for _ in range(len(x))]

        # first

        x_refined = self.refiners[0](x[1])
        x_total = torch.cat((x[0], x_refined), dim=1)
        y[0] = self.convs[0](x_total)

        # all_others

        for i in range(1, self.depth - 1):
            # TO DO: clarify indices
            x_refined = self.refiners[i](x[i + 1])
            x_coarsed = self.coarsers[i - 1](x[i - 1])
            x_total = torch.cat((x_coarsed,
                                 x[i],
                                 x_refined), dim=1)
            y[i] = self.convs[i](x_total)

        # last

        x_coarsed = self.coarsers[-1](x[-2])
        x_total = torch.cat((x_coarsed, x[-1]), dim=1)
        y[-1] = self.convs[-1](x_total)

        return y

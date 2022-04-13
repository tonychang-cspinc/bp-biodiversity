import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifyRegressNet(nn.Module):
    """2 part model - first model classifies the input, then given the classification selects a model to do regression"""

    def __init__(self, classifier, class_model_map):
        super(ClassifyRegressNet, self)
        self.classifier = classifier
        self.class_model_map = class_model_map

    def forward(self, *args):
        class_out = self.classifier(args)
        class_preds = torch.argmax(class_out, dim=1)
        results = []
        for i, pred in enumerate(class_preds):
            inp = [a[i].unsqueeze(0) for a in args]
            results.append(self.class_model_map[pred](inp))
        return class_out, torch.cat(results,dim=0)


class FullNet(nn.Module):
    """Net operating on NAIP, HLS, NASADEM, Daymet. Feeds class output into final regression module."""

    def __init__(self, num_classes=0, num_regressions=1, naip_size=256, hls_size=32):
        super(FullNet, self).__init__()
        self.num_classes = num_classes
        self.num_regressions = num_regressions
        self.naip = DenseStack(stack_size=5, size=naip_size)
        self.hls = HLSBlock(12, 7, hls_size)
        self.climate = Climate()
        self.terrain = Terrain()

        if self.num_classes > 0:
            self.regression = RegressionBlock(64+64+256+16+self.num_classes, num_regressions)
        else:
            self.regression = RegressionBlock(64+64+256+16, num_regressions)
        self.classification = ClassificationBlock(64+64+256+16, num_classes)

    def forward(self, naip_input, hls_input, climate_input, terrain_input):
        climate = self.climate(climate_input)
        naip = self.naip(naip_input)
        hls = self.hls(hls_input)
        terrain = self.terrain(terrain_input)
        ds = torch.cat([naip, hls, climate, terrain], dim=1)
        class_out = None
        regress_out = None
        if self.num_classes > 0:
            class_out = self.classification(ds)
        if self.num_regressions > 0:
            if self.num_classes > 0:
                dsc = torch.cat([ds, class_out], dim=1)
                regress_out = self.regression(dsc)
            else:
                regress_out = self.regression(ds)
        return class_out, regress_out


class NaipNet(nn.Module):
    """Net operating on NAIP. Feeds class output into final regression module."""

    def __init__(self, num_classes=0, num_regressions=1, naip_size=256, hls_size=32):
        super(NaipNet, self).__init__()
        self.num_classes = num_classes
        self.num_regressions = num_regressions
        self.naip = DenseStack(stack_size=5, size=naip_size)
        if self.num_classes > 0:
            self.regression = RegressionBlock(256+self.num_classes, num_regressions)
        else:
            self.regression = RegressionBlock(256, num_regressions)
        self.classification = ClassificationBlock(256, num_classes)

    def forward(self, tensor):
        ds = self.naip(tensor)
        class_out = None
        regress_out = None
        if self.num_classes > 0:
            class_out = self.classification(ds)
        if self.num_regressions > 0:
            if self.num_classes > 0:
                dsc = torch.cat([ds, class_out], dim=1)
                regress_out = self.regression(dsc)
            else:
                regress_out = self.regression(ds)
        return class_out, regress_out


class HLSNet(nn.Module):
    """Net operating on HLS."""

    def __init__(self, num_classes=0, num_regressions=1, naip_size=256, hls_size=32):
        super(HLSNet, self).__init__()
        self.num_classes = num_classes
        self.num_regressions = num_regressions
        self.hls = HLSBlock(12, 7, hls_size)

        # Don't pass classification into regression block for HLS because classification is so bad
        self.regression = RegressionBlock(64, num_regressions)
        self.classification = ClassificationBlock(64, num_classes)

    def forward(self, tensor):
        ds = self.hls(tensor)
        class_out = None
        regress_out = None
        if self.num_classes > 0:
            class_out = self.classification(ds)
        if self.num_regressions > 0:
            regress_out = self.regression(ds)
        return class_out, regress_out


class HLSMergeNet(nn.Module):
    """Net operating on HLS data.

    Differs from HLSNet by the output of classification being fed into the final module for regression
    """

    def __init__(self, num_classes=0, num_regressions=1, naip_size=256, hls_size=32):
        super(HLSMergeNet, self).__init__()
        self.num_classes = num_classes
        self.num_regressions = num_regressions
        self.climate = Climate()
        self.terrain = Terrain()
        self.hls = HLSBlock(12, 7, hls_size)
        if self.num_classes > 0:
            self.regression = RegressionBlock(64+self.num_classes, num_regressions)
        else:
            self.regression = RegressionBlock(64, num_regressions)
        self.classification = ClassificationBlock(64, num_classes)

    def forward(self, hls_tensor):
        h_out = self.hls(hls_tensor)
        class_out = None
        regress_out = None
        if self.num_classes > 0:
            class_out = self.classification(h_out)
        if self.num_regressions > 0:
            if self.num_classes > 0:
                dsc = torch.cat([h_out, class_out], dim=1)
                regress_out = self.regression(dsc)
            else:
                regress_out = self.regression(h_out)
        return class_out, regress_out


class AuxNet(nn.Module):
    """Net operating on daymet + nasadem data. Feeds class output into final regression module."""

    def __init__(self, num_classes=0, num_regressions=1, naip_size=256, hls_size=32):
        super(AuxNet, self).__init__()
        self.num_classes = num_classes
        self.num_regressions = num_regressions
        self.climate = Climate()
        self.terrain = Terrain()
        if self.num_classes > 0:
            self.regression = RegressionBlock(64+16+self.num_classes, num_regressions)
        else:
            self.regression = RegressionBlock(64+16, num_regressions)
        self.classification = ClassificationBlock(64+16, num_classes)

    def forward(self, climate_tensor, terrain_tensor):
        c_out = self.climate(climate_tensor)
        t_out = self.terrain(terrain_tensor)
        ds = torch.cat([c_out, t_out], dim=1)
        class_out = None
        regress_out = None
        if self.num_classes > 0:
            class_out = self.classification(ds)
        if self.num_regressions > 0:
            if self.num_classes > 0:
                dsc = torch.cat([ds, class_out], dim=1)
                regress_out = self.regression(dsc)
            else:
                regress_out = self.regression(ds)
        return class_out, regress_out


class HLSAuxNet(nn.Module):
    """Net operating on hls + daymet + nasadem data. Feeds class output into final regression module."""

    def __init__(self, num_classes=0, num_regressions=1, naip_size=256, hls_size=32):
        super(HLSAuxNet, self).__init__()
        self.num_classes = num_classes
        self.num_regressions = num_regressions
        self.climate = Climate()
        self.terrain = Terrain()
        self.hls = HLSBlock(12, 7, hls_size)
        if self.num_classes > 0:
            self.regression = RegressionBlock(64+64+16+self.num_classes, num_regressions)
        else:
            self.regression = RegressionBlock(64+64+16, num_regressions)
        self.classification = ClassificationBlock(64+64+16, num_classes)

    def forward(self, hls_tensor, climate_tensor, terrain_tensor):
        c_out = self.climate(climate_tensor)
        t_out = self.terrain(terrain_tensor)
        h_out = self.hls(hls_tensor)
        ds = torch.cat([h_out, c_out, t_out], dim=1)
        class_out = None
        regress_out = None
        if self.num_classes > 0:
            class_out = self.classification(ds)
        if self.num_regressions > 0:
            if self.num_classes > 0:
                dsc = torch.cat([ds, class_out], dim=1)
                regress_out = self.regression(dsc)
            else:
                regress_out = self.regression(ds)
        return class_out, regress_out


class SatNet(nn.Module):
    """Net operating on satellite imagery (NAIP + HLS). Feeds class output into final regression module."""

    def __init__(self, num_classes=0, num_regressions=1, naip_size=256, hls_size=32):
        super(SatNet, self).__init__()
        self.num_classes = num_classes
        self.num_regressions = num_regressions
        self.hls = HLSBlock(12, 7, hls_size)
        self.naip = DenseStack(stack_size=5, size=naip_size)
        if self.num_classes > 0:
            self.regression = RegressionBlock(64+256+self.num_classes, num_regressions)
        else:
            self.regression = RegressionBlock(64+256, num_regressions)
        self.classification = ClassificationBlock(64+256, num_classes)

    def forward(self, naip_tensor, hls_tensor):
        n_out = self.naip(naip_tensor)
        h_out = self.hls(hls_tensor)
        ds = torch.cat([n_out, h_out], dim=1)
        class_out = None
        regress_out = None
        if self.num_classes > 0:
            class_out = self.classification(ds)
        if self.num_regressions > 0:
            if self.num_classes > 0:
                dsc = torch.cat([ds, class_out], dim=1)
                regress_out = self.regression(dsc)
            else:
                regress_out = self.regression(ds)
        return class_out, regress_out


class Terrain(nn.Module):
    """ Input is a tensor with elevation, slope, aspect
        Input shape: N, 3
        Output shape: N, 16
    """
    def __init__(self):
        super(Terrain, self).__init__()
        self.terrain = nn.Sequential(
            nn.Linear(3,16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

    def forward(self, tensor):
        return self.terrain(tensor)


class Climate(nn.Module):
    """ Input is a tensor with 12 months of monthly means + stds over 40 years for 5 climate variables
        Input shape: N, 2 (means + stds), 5 (prcp, tmax, tmin, vp, swe), 12 (months)
        Output shape: N, 64
    """
    def __init__(self):
        super(Climate, self).__init__()
        self.climate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 5 * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, tensor):
        return self.climate(tensor)


class DenseStack(nn.Module):
    """A stack of DenseBlock -> TransitionBlock -> DenseBlock -> ...
    Number of DenseBlocks is n+1

    Input shape: N, input_channels, size, size
    Output shape: N, output_size
    """
    def __init__(
        self,
        stack_size=1,
        input_channels=3,
        n_filters_1=16,
        n_filters_2=12,
        output_size=256,
        dropout=0.1,
        weight_decay=1E-4,
        size=256
    ):
        super(DenseStack, self).__init__()
        stack = [
            nn.Conv2d(input_channels, n_filters_1, 1, bias=False)
        ]
        trans_out_filters = n_filters_1
        for n in range(stack_size):
            dense_in_filters = trans_out_filters
            trans_in_filters = dense_in_filters + n_filters_2 * 2
            trans_out_filters = n_filters_1 * 2 * (n + 1)
            stack.extend([
                DenseBlock(dense_in_filters, n_filters_2, dropout, weight_decay),
                TransitionBlock(trans_in_filters, trans_out_filters, dropout, weight_decay)
            ])
        final_channels = trans_out_filters + n_filters_2 * 2
        output_dim = int(size / (2**stack_size))
        stack.extend([
            DenseBlock(trans_out_filters, n_filters_2, dropout, weight_decay),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(),
            # Equivalent to GlobalAveragePooling2D in tf
            nn.AvgPool2d(output_dim),
            nn.Flatten(),
            nn.Linear(final_channels, output_size),
            nn.ReLU()
        ])
        self.stack = nn.Sequential(*stack)

    def forward(self, tensor):
        return self.stack(tensor)


class DenseBlock(nn.Module):
    """
        Input shape: N, input_filters, H, W
        Output shape: N, input_filters + n_filters * 2, H, W
    """
    def __init__(self, input_filters=16, n_filters=12, dropout=0.1, weight_decay=1E-4):
        super(DenseBlock, self).__init__()

        self.dense1 = nn.Sequential(
            nn.BatchNorm2d(input_filters),
            nn.ReLU(),
            nn.Conv2d(input_filters, n_filters, 3, padding=1, bias=False),
            nn.Dropout(dropout),
        )

        self.dense2 = nn.Sequential(
            nn.BatchNorm2d(n_filters + input_filters),
            nn.ReLU(),
            nn.Conv2d(input_filters+n_filters, n_filters, 3, padding=1, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, tensor):
        d1_out = self.dense1(tensor)
        d2_in = torch.cat([tensor, d1_out], dim=1)
        d2_out = self.dense2(d2_in)

        return torch.cat([tensor, d1_out, d2_out], dim=1)


class TransitionBlock(nn.Module):
    """
        Input shape: N, input_filters, H, W
        Output shape: N, n_filters, H/2, W/2
    """
    def __init__(self, input_filters, n_filters, dropout=0.1, weight_decay=1E-4):
        super(TransitionBlock, self).__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(input_filters),
            nn.ReLU(),
            nn.Conv2d(input_filters, n_filters, 1, bias=False),
            nn.Dropout(dropout),
            nn.AvgPool2d(2, 2)
        )

    def forward(self, tensor):
        return self.block(tensor)


class HLSBlock(nn.Module):
    """
        Input shape: N, seq_len (1 image for each month), n_filters, size, size
        Output shape: N, 64
    """
    def __init__(self, seq_len, n_filters, size):
        super(HLSBlock, self).__init__()
        self.block = nn.Sequential(
            TimeDistributed(nn.Conv2d(n_filters, 64, 3, padding=1, bias=False)),
            TimeDistributed(nn.Conv2d(64, 64, 3, padding=1, bias=False)),
            TimeDistributed(nn.Conv2d(64, 128, 3, padding=1, bias=False)),
            TimeDistributed(nn.Flatten()),
            nn.Dropout(0.1),
            nn.LSTM(128*(size**2), 64, batch_first=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*seq_len, 64)
        )

    def forward(self, tensor):
        block_y, _ = self.block(tensor)
        return self.fc(block_y)


class TimeDistributed(nn.Module):
    """ Adapted from information in
        https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/2.
    """
    def __init__(self, module):
        """
            Input shape: N, n_images, n_channels, H, W
            Output shape: N, n_images, module_output_shape
        """
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, tensor):

        # Squash batch samples (index 0) and timesteps (index 1) into a single axis
        reshaped = tensor.view(-1, *tensor.shape[2:])
        y = self.module(reshaped)

        # Reshape y back to (N, n_images, ...)
        return y.view(tensor.shape[0], tensor.shape[1], *y.shape[1:])


class RegressionBlock(nn.Module):
    def __init__(self, n_dims, n_outputs):
        """
            Input shape: N, n_dims
            Output shape: N, n_outputs
        """
        super(RegressionBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(n_dims, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, n_outputs),
            nn.ReLU()
        )

    def forward(self, tensor):
        return self.block(tensor)


class ClassificationBlock(nn.Module):
    def __init__(self, n_dims, n_outputs):
        """
            Input shape: N, n_dims
            Output shape: N, n_outputs
        """
        super(ClassificationBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(n_dims, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, n_outputs),
            nn.Softmax(dim=1)
        )

    def forward(self, tensor):
        return self.block(tensor)

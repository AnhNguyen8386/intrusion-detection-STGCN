import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Graph():
    def __init__(self, **kwargs):
        self.A = self.get_adjacency_matrix(kwargs.get('strategy', 'spatial'))
    
    def get_adjacency_matrix(self, strategy):
        if strategy == 'spatial':
            num_joint = 17
            self.num_joint = num_joint

            self.edges = [
                (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
                (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
                (11, 13), (13, 15), (12, 14), (14, 16)
            ]

            A = np.zeros((num_joint, num_joint))
            for i, j in self.edges:
                A[i, j] = 1
                A[j, i] = 1
            
            I = np.eye(num_joint)
            A = A + I
            
            return torch.tensor(A, dtype=torch.float32)

        raise ValueError("Unsupported strategy")

class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(t_kernel_size, 1), padding=(t_padding, 0), stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x, A):
        N, C, T, V = x.size()
        x = self.conv(x)
        x = x.view(N, self.kernel_size, x.size(1) // self.kernel_size, T, V)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()

class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = GraphConvolution(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)

class StreamSpatialTemporalGraph(nn.Module):
    def __init__(self, in_channels, graph_args, num_class=None, edge_importance_weighting=True, **kwargs):
        super().__init__()
        graph = Graph(**graph_args)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs)
        ))
        
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(A.size(0), A.size(1), A.size(1))) for _ in self.st_gcn_networks])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        if num_class is not None:
            self.cls = nn.Conv2d(256, num_class, kernel_size=1)
        else:
            self.cls = lambda x: x

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        
        A_final = self.A.unsqueeze(0).repeat(self.A.size(0), 1, 1)
        
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, A_final * importance)
            
        if hasattr(self, 'cls'):
            x = F.avg_pool2d(x, x.size()[2:])
            x = self.cls(x).view(x.size(0), -1)
        return x

class EnhancedMultimodalGraph(nn.Module):
    def __init__(self, graph_args, num_class, edge_importance_weighting=True, **kwargs):
        super().__init__()
        
        self.pts_stream = StreamSpatialTemporalGraph(3, graph_args, None, edge_importance_weighting, **kwargs)
        self.mot_stream = StreamSpatialTemporalGraph(2, graph_args, None, edge_importance_weighting, **kwargs)
        
        self.visual_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.bbox_fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.fcn = nn.Sequential(
            nn.Linear(256 + 256 + 128 + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_class)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs):
        pose_data, motion_data, visual_data, bbox_data = inputs
        
        pose_out = self.pts_stream(pose_data)
        motion_out = self.mot_stream(motion_data)
        pose_motion_concat = torch.cat([pose_out, motion_out], dim=-1)
        
        visual_pooled = torch.mean(visual_data, dim=1)
        visual_out = self.visual_fc(visual_pooled)
        
        bbox_pooled = torch.mean(bbox_data, dim=1)
        bbox_out = self.bbox_fc(bbox_pooled)
        
        concat_features = torch.cat([pose_motion_concat, visual_out, bbox_out], dim=-1)
        out = self.fcn(concat_features)
        
        return out
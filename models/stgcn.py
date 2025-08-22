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
              
            # Create base adjacency matrix  
            A = np.zeros((num_joint, num_joint))  
            for i, j in self.edges:  
                A[i, j] = 1  
                A[j, i] = 1  
            I = np.eye(num_joint)  
            A = A + I  
              
            # Normalize adjacency matrix for better gradient flow  
            D = np.sum(A, axis=1)  
            D = np.diag(np.power(D + 1e-6, -0.5))  
            A = D @ A @ D  
              
            # Create proper 3D spatial partitions for better spatial modeling  
            A_spatial = []  
            A_root = np.eye(num_joint)  # Self-connections  
            A_neighbor = A - A_root     # Neighbor connections  
              
            A_spatial.append(A_root)  
            A_spatial.append(A_neighbor)  
              
            # Stack to create (K, V, V) format where K=2  
            return torch.tensor(np.stack(A_spatial), dtype=torch.float32)  
              
        raise ValueError("Unsupported strategy")  
  
class GraphConvolution(nn.Module):  
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):  
        super().__init__()  
        self.kernel_size = kernel_size  
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size,   
                             kernel_size=(t_kernel_size, 1), padding=(t_padding, 0),   
                             stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)  
          
        # Better weight initialization for stability  
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')  
        if self.conv.bias is not None:  
            nn.init.constant_(self.conv.bias, 0)  
  
    def forward(self, x, A):  
        # Handle both 2D and 3D adjacency matrices robustly  
        if A.dim() == 2:  
            A = A.unsqueeze(0)  
          
        N, C, T, V = x.size()  
        x = self.conv(x)  
        x = x.reshape(N, self.kernel_size, x.size(1) // self.kernel_size, T, V)  
        x = torch.einsum('nkctv,kvw->nctw', (x, A))  
        return x.contiguous()  
  
class st_gcn(nn.Module):  
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.6, residual=True):  
        super().__init__()  
        assert len(kernel_size) == 2  
        assert kernel_size[0] % 2 == 1  
        padding = ((kernel_size[0] - 1) // 2, 0)  
          
        self.gcn = GraphConvolution(in_channels, out_channels, kernel_size[1])  
          
        # Balanced TCN with moderate regularization  
        self.tcn = nn.Sequential(  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=False),  
            nn.Dropout(dropout * 0.4, inplace=False),  # Reduced first dropout  
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),  
            nn.BatchNorm2d(out_channels),  
            nn.Dropout(dropout * 0.6, inplace=False)  # Moderate second dropout  
        )  
          
        if not residual:  
            self.residual = lambda x: 0  
        elif (in_channels == out_channels) and (stride == 1):  
            self.residual = lambda x: x  
        else:  
            self.residual = nn.Sequential(  
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),  
                nn.BatchNorm2d(out_channels),  
                nn.Dropout(dropout * 0.2, inplace=False)  # Light residual dropout  
            )  
        self.relu = nn.ReLU(inplace=False)  
  
    def forward(self, x, A):  
        res = self.residual(x)  
        x = self.gcn(x, A)  
        x = self.tcn(x) + res  
        return self.relu(x)  
  
class StreamSpatialTemporalGraph(nn.Module):  
    def __init__(self, in_channels, graph_args, num_class=None, edge_importance_weighting=True, dropout=0.6, **kwargs):  
        super().__init__()  
          
        # Load graph with proper 3D adjacency matrix  
        graph = Graph(**graph_args)  
        A = graph.A.clone().detach().requires_grad_(False)  
        self.register_buffer('A', A)  
          
        spatial_kernel_size = A.size(0)  # Now properly 2 (root + neighbor)  
        temporal_kernel_size = 9  
        kernel_size = (temporal_kernel_size, spatial_kernel_size)  
          
        # Use Identity to avoid BatchNorm dimension issues  
        self.data_bn = nn.Identity()  
          
        # Optimized architecture with balanced regularization  
        self.st_gcn_networks = nn.ModuleList((  
            st_gcn(in_channels, 32, kernel_size, 1, dropout=dropout, residual=False, **kwargs),  
            st_gcn(32, 64, kernel_size, 2, dropout=dropout, **kwargs),  
            st_gcn(64, 128, kernel_size, 2, dropout=dropout, **kwargs),  
            st_gcn(128, 256, kernel_size, 1, dropout=dropout, **kwargs)  
            # 4 layers with progressive channel increase  
        ))  
          
        # Initialize edge importance weighting  
        if edge_importance_weighting:  
            self.edge_importance = nn.ParameterList([  
                nn.Parameter(torch.ones(A.size()) * 0.2)  # Slightly higher initialization  
                for _ in self.st_gcn_networks  
            ])  
        else:  
            self.edge_importance = [1] * len(self.st_gcn_networks)  
          
        # Balanced classification head  
        if num_class is not None:  
            self.cls = nn.Sequential(  
                nn.Dropout(dropout * 0.8),  # Moderate dropout  
                nn.Conv2d(256, 128, kernel_size=1),  
                nn.BatchNorm2d(128),  
                nn.ReLU(inplace=False),  
                nn.Dropout(dropout * 0.6),  # Reduced dropout  
                nn.Conv2d(128, num_class, kernel_size=1)  
            )  
        else:  
            self.cls = lambda x: x  
  
    def forward(self, x):  
    # Handle input format conversion  
        if x.dim() == 4 and x.size(-1) in [2, 3]:  # (N, T, V, C)  
            N, T, V, C = x.size()  
            x = x.permute(0, 3, 1, 2).contiguous()  # (N, C, T, V)  
      
        N, C, T, V = x.size()  
      
    # FIXED: Remove keyword argument syntax  
        x = self.data_bn(x)  # Instead of x = self.data_bn(x = x.reshape(...))  
      
    # Forward through ST-GCN layers  
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):  
            if isinstance(importance, nn.Parameter):  
                importance_clamped = torch.clamp(importance, min=0.05, max=1.5)  
                x = gcn(x, self.A * importance_clamped)  
            else:  
                x = gcn(x, self.A)  
      
    # Global average pooling and classification  
        x = F.avg_pool2d(x, x.size()[2:])  
        x = self.cls(x)  
        x = x.reshape(x.size(0), -1)  
      
        return x
  
class TwoStreamSpatialTemporalGraph(nn.Module):  
    def __init__(self, graph_args, num_class, edge_importance_weighting=True, dropout=0.6, **kwargs):  
        super().__init__()  
          
        # Both streams use 3 channels to match training data format  
        self.pts_stream = StreamSpatialTemporalGraph(3, graph_args, None,   
                                                   edge_importance_weighting, dropout, **kwargs)  
        self.mot_stream = StreamSpatialTemporalGraph(3, graph_args, None,   
                                                   edge_importance_weighting, dropout, **kwargs)  
          
        # Balanced fusion layer with moderate regularization  
        self.fcn = nn.Sequential(  
            nn.Dropout(dropout * 0.7),  # Moderate input dropout  
            nn.Linear(256 * 2, 256),  
            nn.BatchNorm1d(256),  
            nn.ReLU(inplace=False),  
            nn.Dropout(dropout * 0.5),  # Reduced middle dropout  
            nn.Linear(256, 128),  
            nn.BatchNorm1d(128),  
            nn.ReLU(inplace=False),  
            nn.Dropout(dropout * 0.3),  # Light final dropout  
            nn.Linear(128, num_class)  
        )  
          
        # Initialize fusion layer weights properly  
        self._initialize_fusion_weights()  
  
    def _initialize_fusion_weights(self):  
        """Initialize fusion layer weights for better convergence"""  
        for m in self.fcn.modules():  
            if isinstance(m, nn.Linear):  
                nn.init.xavier_uniform_(m.weight, gain=0.2)  # Balanced gain  
                if m.bias is not None:  
                    nn.init.constant_(m.bias, 0.01)  
            elif isinstance(m, nn.BatchNorm1d):  
                nn.init.constant_(m.weight, 1)  
                nn.init.constant_(m.bias, 0)  
  
    def forward(self, inputs):  
        out1 = self.pts_stream(inputs[0])  
        out2 = self.mot_stream(inputs[1])  
          
        concat = torch.cat([out1, out2], dim=-1)  
        out = self.fcn(concat)  
          
        return torch.sigmoid(out)  # BCE loss compatible  
  
# Enhanced Multimodal Graph for future extensions  
class EnhancedMultimodalGraph(nn.Module):  
    def __init__(self, graph_args, num_classes, dropout=0.6):  
        super().__init__()  
          
        self.st_gcn_pose = StreamSpatialTemporalGraph(  
            in_channels=3,  
            graph_args=graph_args,  
            num_class=None,  
            edge_importance_weighting=True,  
            dropout=dropout  
        )  
  
        self.st_gcn_motion = StreamSpatialTemporalGraph(  
            in_channels=2,  
            graph_args=graph_args,  
            num_class=None,  
            edge_importance_weighting=True,  
            dropout=dropout  
        )  
  
        # Visual stream for ResNet18 features (512 dimensions)  
        self.visual_stream = nn.Sequential(  
            nn.Linear(512, 256),  
            nn.BatchNorm1d(256),  
            nn.ReLU(inplace=False),  
            nn.Dropout(dropout),  
            nn.Linear(256, 128),  
            nn.BatchNorm1d(128),  
            nn.ReLU(inplace=False),  
            nn.Dropout(dropout * 0.7)  # Reduced dropout  
        )  
          
        self.bbox_stream = nn.Sequential(  
            nn.Linear(4, 64),  
            nn.BatchNorm1d(64),  
            nn.ReLU(inplace=False),  
            nn.Dropout(dropout * 0.8),  
            nn.Linear(64, 32),  
            nn.BatchNorm1d(32),  
            nn.ReLU(inplace=False),  
            nn.Dropout(dropout * 0.6)  
        )  
  
        self.scene_stream = nn.Sequential(  
            nn.Linear(512, 256),  
            nn.BatchNorm1d(256),  
            nn.ReLU(inplace=False),  
            nn.Dropout(dropout),  
            nn.Linear(256, 128),  
            nn.BatchNorm1d(128),  
            nn.ReLU(inplace=False),  
            nn.Dropout(dropout * 0.7)  
        )  
  
        # Fusion layer  
        fusion_input_dim = 256 + 256 + 128 + 32 + 128  # pose + motion + visual + bbox + scene  
          
        self.fusion_layer = nn.Sequential(  
            nn.Linear(fusion_input_dim, 512),  
            nn.BatchNorm1d(512),  
            nn.ReLU(inplace=False),  
            nn.Dropout(dropout * 0.8),  
            nn.Linear(512, 256),  
            nn.BatchNorm1d(256),  
            nn.ReLU(inplace=False),  
            nn.Dropout(dropout * 0.6),  
            nn.Linear(256, num_classes)  
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
  
    def forward(self, data):  
        # Backward compatibility  
        if len(data) == 4:  
            pose, mot, visual, bbox = data  
            scene = torch.zeros_like(visual)  
        else:  
            pose, mot, visual, bbox, scene = data  
              
        pose_features = self.st_gcn_pose(pose)  
        mot_features = self.st_gcn_motion(mot)  
          
        # Process visual features  
        N_visual, T_visual, F_visual = visual.shape  
        visual_features = visual.reshape(-1, F_visual)  
        visual_features = self.visual_stream(visual_features)  
        visual_features = visual_features.reshape(N_visual, T_visual, -1)  
        visual_features = torch.mean(visual_features, dim=1)  
  
        # Process bbox features  
        N_bbox, T_bbox, F_bbox = bbox.shape  
        bbox_features = bbox.reshape(-1, F_bbox)  
        bbox_features = self.bbox_stream(bbox_features)  
        bbox_features = bbox_features.reshape(N_bbox, T_bbox, -1)  
        bbox_features = torch.mean(bbox_features, dim=1)  
  
        # Process scene features  
        N_scene, T_scene, F_scene = scene.shape  
        scene_features = scene.reshape(-1, F_scene)  
        scene_features = self.scene_stream(scene_features)  
        scene_features = scene_features.reshape(N_scene, T_scene, -1)  
        scene_features = torch.mean(scene_features, dim=1)  
  
        # Fusion  
        fused_features = torch.cat((pose_features, mot_features, visual_features, bbox_features, scene_features), dim=1)  
        return self.fusion_layer(fused_features)  
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Dict

class SpatialProjectionHead(nn.Module):
    """
    Spatial-aware projection head that preserves spatial structure
    """
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        
        # Convolutional projection that maintains spatial dimensions
        self.conv_proj = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 2, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_proj(x)

class MultiScaleFeatureExtractor(nn.Module):
    """
    latent features at multiple spatial scales from ResNet backbone
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        
        # Split ResNet into stages for multi-scale extraction
        layers = list(backbone.children())
        
        self.stem = nn.Sequential(*layers[:4])
        
        # ResNet stages
        self.stage1 = layers[4]  # layer1 
        self.stage2 = layers[5]  # layer2 
        self.stage3 = layers[6]  # layer3 
        self.stage4 = layers[7]  # layer4 
        
        # Feature dimensions at each stage
        self.feature_dims = {
            'stage1': 256,
            'stage2': 512,
            'stage3': 1024,
            'stage4': 2048
        }
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        
        feat1 = self.stage1(x)  # Fine details
        feat2 = self.stage2(feat1)  # Medium structures
        feat3 = self.stage3(feat2)  # Larger patterns
        feat4 = self.stage4(feat3)  # Global context
        
        return {
            'stage1': feat1,
            'stage2': feat2,
            'stage3': feat3,
            'stage4': feat4
        }

class SpatialPredictionHead(nn.Module):
    """
    Dense prediction head that operates on spatial feature maps
    """
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)

class SpatialBYOL(nn.Module):
    def __init__(
        self,
        encoder_arch: str = 'resnet50',
        pretrained: bool = True,
        projection_dim: int = 256,
        spatial_projection_dim: int = 128,
        hidden_dim: int = 4096,
        tau: float = 0.996,
        use_multi_scale: bool = True,
        spatial_loss_weight: float = 0.5
    ):
        super().__init__()
        
        self.tau = tau
        self.use_multi_scale = use_multi_scale
        self.spatial_loss_weight = spatial_loss_weight
        
        # Load pretrained ResNet
        if encoder_arch == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
        elif encoder_arch == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported architecture: {encoder_arch}")
        
        # latent feature extractor 
        self.online_encoder = MultiScaleFeatureExtractor(backbone)
        
        # Global branch (standard BYOL)
        self.online_global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.online_global_projection = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.online_global_prediction = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Spatial branch
        self.online_spatial_projection = SpatialProjectionHead(1024, spatial_projection_dim)
        self.online_spatial_prediction = SpatialPredictionHead(
            spatial_projection_dim, 
            spatial_projection_dim * 2, 
            spatial_projection_dim
        )
        
        # Multi-scale fusion (if enabled)
        # if use_multi_scale:
        #     self.scale_fusion = nn.ModuleDict({
        #         'stage2': nn.Conv2d(512, spatial_projection_dim, kernel_size=1),
        #         'stage3': nn.Conv2d(1024, spatial_projection_dim, kernel_size=1),
        #         'stage4': nn.Conv2d(2048, spatial_projection_dim, kernel_size=1)
        #     })
        

        # TARGET NETWORK
        
        # Create target network with same architecture
        target_backbone = models.resnet50(pretrained=pretrained) if encoder_arch == 'resnet50' else models.resnet101(pretrained=pretrained)
        self.target_encoder = MultiScaleFeatureExtractor(target_backbone)
        
        # Target global branch
        self.target_global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.target_global_projection = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Target spatial branch
        self.target_spatial_projection = SpatialProjectionHead(1024, spatial_projection_dim)
        
        # Initialize target network
        self._initialize_target_network()
        self._set_target_requires_grad(False)
        
        print(f"  Spatial-BYOL initialized")
        print(f"  Global projection: {projection_dim}D")
        print(f"  Spatial projection: {spatial_projection_dim}D")
        print(f"  Multi-scale: {use_multi_scale}")
        print(f"  Spatial loss weight: {spatial_loss_weight}")
    
    def _initialize_target_network(self):
        """Copy online network weights to target network."""
        # Encoder
        for online_param, target_param in zip(
            self.online_encoder.parameters(), 
            self.target_encoder.parameters()
        ):
            target_param.data.copy_(online_param.data)
        
        # Global projection
        for online_param, target_param in zip(
            self.online_global_projection.parameters(),
            self.target_global_projection.parameters()
        ):
            target_param.data.copy_(online_param.data)
        
        # Spatial projection
        for online_param, target_param in zip(
            self.online_spatial_projection.parameters(),
            self.target_spatial_projection.parameters()
        ):
            target_param.data.copy_(online_param.data)
    
    def _set_target_requires_grad(self, requires_grad: bool):
        """Control gradient computation for target network."""
        for module in [self.target_encoder, self.target_global_projection, 
                       self.target_spatial_projection]:
            for param in module.parameters():
                param.requires_grad = requires_grad
    
    @torch.no_grad()
    def update_target_network(self):
        """EMA update of target network."""
        # Encoder
        for online_param, target_param in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_param.data = self.tau * target_param.data + (1 - self.tau) * online_param.data
        
        # Global projection
        for online_param, target_param in zip(
            self.online_global_projection.parameters(),
            self.target_global_projection.parameters()
        ):
            target_param.data = self.tau * target_param.data + (1 - self.tau) * online_param.data
        
        # Spatial projection
        for online_param, target_param in zip(
            self.online_spatial_projection.parameters(),
            self.target_spatial_projection.parameters()
        ):
            target_param.data = self.tau * target_param.data + (1 - self.tau) * online_param.data
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[Dict, Dict]:  
        """Forward pass through online and target networks."""
        
        # Extract multi-scale features
        online_feats_1 = self.online_encoder(x1)
        online_feats_2 = self.online_encoder(x2)
        
        # Global branch (standard BYOL)
        global_feat_1 = self.online_global_pool(online_feats_1['stage4'])
        global_feat_2 = self.online_global_pool(online_feats_2['stage4'])
        global_feat_1 = torch.flatten(global_feat_1, 1)
        global_feat_2 = torch.flatten(global_feat_2, 1)
        
        global_proj_1 = self.online_global_projection(global_feat_1)
        global_proj_2 = self.online_global_projection(global_feat_2)
        
        global_pred_1 = self.online_global_prediction(global_proj_1)
        global_pred_2 = self.online_global_prediction(global_proj_2)
        
        # Spatial branch (new for OCT)
        spatial_proj_1 = self.online_spatial_projection(online_feats_1['stage3'])
        spatial_proj_2 = self.online_spatial_projection(online_feats_2['stage3'])
        
        spatial_pred_1 = self.online_spatial_prediction(spatial_proj_1)
        spatial_pred_2 = self.online_spatial_prediction(spatial_proj_2)

        # TARGET NETWORK        
        with torch.no_grad():
            # Extract features
            target_feats_1 = self.target_encoder(x1)
            target_feats_2 = self.target_encoder(x2)
            
            # Global branch
            global_feat_1_t = self.target_global_pool(target_feats_1['stage4'])
            global_feat_2_t = self.target_global_pool(target_feats_2['stage4'])
            global_feat_1_t = torch.flatten(global_feat_1_t, 1)
            global_feat_2_t = torch.flatten(global_feat_2_t, 1)
            
            global_proj_1_t = self.target_global_projection(global_feat_1_t)
            global_proj_2_t = self.target_global_projection(global_feat_2_t)
            
            # Spatial branch
            spatial_proj_1_t = self.target_spatial_projection(target_feats_1['stage3'])
            spatial_proj_2_t = self.target_spatial_projection(target_feats_2['stage3'])
        
        # combined outputs
        online_outputs = {
            'global_pred_1': global_pred_1,
            'global_pred_2': global_pred_2,
            'spatial_pred_1': spatial_pred_1,
            'spatial_pred_2': spatial_pred_2
        }
        
        target_outputs = {
            'global_proj_1': global_proj_1_t,
            'global_proj_2': global_proj_2_t,
            'spatial_proj_1': spatial_proj_1_t,
            'spatial_proj_2': spatial_proj_2_t
        }
        
        return online_outputs, target_outputs
    
    def get_encoder(self) -> nn.Module:
        """Return encoder for downstream tasks."""
        return self.online_encoder

def spatial_byol_loss(
    online_outputs: Dict[str, torch.Tensor],
    target_outputs: Dict[str, torch.Tensor],
    spatial_weight: float = 0.5
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """ Hybrid loss  """
    
    def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity loss."""
        pred_norm = F.normalize(pred, dim=-1, p=2)
        target_norm = F.normalize(target, dim=-1, p=2)
        return 2 - 2 * (pred_norm * target_norm).sum(dim=-1).mean()
    
    def spatial_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity loss for spatial feature maps.
        """
        # Normalize along channel dimension
        pred_norm = F.normalize(pred, dim=1, p=2)
        target_norm = F.normalize(target, dim=1, p=2)
        
        # Compute cosine similarity at each spatial location
        similarity = (pred_norm * target_norm).sum(dim=1)
        return 2 - 2 * similarity.mean()
    
    
    global_loss_1 = cosine_loss(
        online_outputs['global_pred_1'],
        target_outputs['global_proj_2']
    )
    global_loss_2 = cosine_loss(
        online_outputs['global_pred_2'],
        target_outputs['global_proj_1']
    )
    global_loss = (global_loss_1 + global_loss_2) / 2
    
    
    spatial_loss_1 = spatial_cosine_loss(
        online_outputs['spatial_pred_1'],
        target_outputs['spatial_proj_2']
    )
    spatial_loss_2 = spatial_cosine_loss(
        online_outputs['spatial_pred_2'],
        target_outputs['spatial_proj_1']
    )
    spatial_loss = (spatial_loss_1 + spatial_loss_2) / 2
    
    total_loss = (1 - spatial_weight) * global_loss + spatial_weight * spatial_loss
    
    # Return loss
    loss_dict = {
        'total': total_loss.item(),
        'global': global_loss.item(),
        'spatial': spatial_loss.item()
    }
    
    return total_loss, loss_dict


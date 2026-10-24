import torch
import torch.nn as nn
import torchvision.models as models

# =====================================================================
# STAGE 1: Spatial Feature Extraction
# Extracts visual features from individual video frames (B, C, T, H, W) -> (B, T, D)
# =====================================================================
class Stage1_SpatialExtractor(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Output: (B*T, 512, 1, 1)
        self.projection = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # Input tensor shape: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        
        feats = self.backbone(x_reshaped).flatten(1)       # (B*T, 512)
        feats = self.projection(feats)                     # (B*T, feature_dim)
        
        # Reshape to sequence representation: (B, T, feature_dim)
        return feats.view(B, T, -1)


# =====================================================================
# STAGE 2: Temporal Feature Extraction
# Tracks motion and contraction changes across consecutive frames
# =====================================================================
class Stage2_TemporalExtractor(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=3,
            padding=1,
            groups=feature_dim
        )
        self.act = nn.GELU()

    def forward(self, x):
        # Input shape: (B, T, feature_dim)
        x_t = x.transpose(1, 2)  # (B, feature_dim, T)
        out = self.act(self.depthwise_conv(x_t))
        return out.transpose(1, 2)  # (B, T, feature_dim)


# =====================================================================
# STAGE 3: Spatial-Temporal Feature Learning
# Fuses visual structure features with whole-sequence motion dynamics
# =====================================================================
class Stage3_SpatiotemporalFusion(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            batch_first=True
        )

    def forward(self, spatial_feats, temporal_feats):
        # Combine spatial and temporal representations
        combined = self.norm(spatial_feats + temporal_feats)
        return self.encoder_layer(combined)  # (B, T, feature_dim)


# =====================================================================
# STAGE 4: Video-Level Feature Aggregation
# Aggregates frame sequence features into a single 1D video vector via Attention
# =====================================================================
class Stage4_VideoAggregation(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Input shape: (B, T, feature_dim)
        attn_logits = self.attn(x)                       # (B, T, 1)
        attn_weights = torch.softmax(attn_logits, dim=1) # (B, T, 1)
        
        video_vector = torch.sum(x * attn_weights, dim=1)  # (B, feature_dim)
        return video_vector, attn_weights


# =====================================================================
# STAGE 5: Cardiac Function Prediction
# Predicts scalar EF (%) as well as auxiliary EDV and ESV targets
# =====================================================================
class Stage5_CardiacPrediction(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.ef_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        self.edv_head = nn.Linear(feature_dim, 1)
        self.esv_head = nn.Linear(feature_dim, 1)

    def forward(self, video_vector):
        ef = self.ef_head(video_vector).squeeze(-1)
        edv = self.edv_head(video_vector).squeeze(-1)
        esv = self.esv_head(video_vector).squeeze(-1)
        
        return {
            'EF': ef,
            'EDV': edv,
            'ESV': esv
        }A
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.distributions import Categorical
from .pixel_coordinate_utils import crop_feature_maps_at_coordinates, normalize_pixel_coordinates

def square_distance(xyz, center_xyz):
    B, N, _ = center_xyz.shape
    _, M, _ = xyz.shape
    dist = -2 * torch.matmul(center_xyz, xyz.permute(0, 2, 1))
    dist += torch.sum(center_xyz ** 2, -1).view(B, N, 1)
    dist += torch.sum(xyz ** 2, -1).view(B, 1, M)
    return dist

def knn_point(neighbor, xyz, center_xyz):
    sqrdists = square_distance(xyz, center_xyz)
    _, group_idx = torch.topk(sqrdists, neighbor, dim = -1, largest=False, sorted=False)
    return group_idx

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def sample_and_group(npoint, radius, neighbor, xyz, feature):
    feature = feature.permute(0, 2, 1)
    B, N, C = xyz.shape     
    S = npoint 
    
    xyz = xyz.contiguous()

    noise = torch.rand(B, N, device=xyz.device) 
    ids_shuffle = torch.argsort(noise, dim=1) 
    ids_keep = ids_shuffle[:, :S]  
    fps_idx = torch.arange(N, dtype=torch.long).to(xyz.device).unsqueeze(0).repeat(B,1)
    fps_idx = torch.gather(fps_idx, dim=1, index=ids_keep)  

    center_xyz = index_points(xyz, fps_idx) 
    center_feature = index_points(feature, fps_idx)  

    idx = knn_point(neighbor, xyz, center_xyz)  
    grouped_feature = index_points(feature, idx)           
    grouped_feature_center = grouped_feature - center_feature.view(B, S, 1, -1) 
    res_points = torch.cat([grouped_feature_center, center_feature.view(B, S, 1, -1).repeat(1, 1, neighbor, 1)], dim=-1)
    return center_xyz, res_points

class Local_op(nn.Module):
   def __init__(self, in_channels, out_channels):
       super(Local_op, self).__init__()
       self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
       self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
       self.bn1 = nn.BatchNorm1d(out_channels)
       self.bn2 = nn.BatchNorm1d(out_channels)

   def forward(self, x):
       b, n, s, d = x.size()
       x = x.permute(0, 1, 3, 2)
       x = x.reshape(-1, d, s)
       batch_size, _, N = x.size()
       x1 = F.relu(self.bn1(self.conv1(x)))
       x2 = F.relu(self.bn2(self.conv2(x1)))
       x3 = F.adaptive_max_pool1d(x2, 1)
       x4 = x3.view(batch_size, -1)
       x_res = x4.reshape(b, n, -1).permute(0, 2, 1)
       return x_res

class ViewEncoder(nn.Module):
    def __init__(self, point_num=8192, feature_dim=128, crop_size=11):
        super(ViewEncoder, self).__init__()
        self.crop_size = crop_size
        
        self.rgb_encoder = models.resnet18(pretrained=True)
        in_features = self.rgb_encoder.fc.in_features
        self.rgb_encoder.fc = nn.Linear(in_features, feature_dim)
        
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 1024, kernel_size=1, stride=int(point_num/256), bias=False)
        self.bn2 = nn.BatchNorm1d(1024)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.conv_fuse1 = nn.Sequential(
            nn.Conv1d(1280, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv1d(512, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.patch_conv = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128)
        )
        
        self.patch_dim = 128
        
        self.patch_attention = nn.MultiheadAttention(
            embed_dim=self.patch_dim,
            num_heads=4,
            batch_first=True
        )
        
        self.patch_norm1 = nn.LayerNorm(self.patch_dim)
        self.patch_norm2 = nn.LayerNorm(self.patch_dim)
        
        self.patch_ffn = nn.Sequential(
            nn.Linear(self.patch_dim, self.patch_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.patch_dim * 2, self.patch_dim)
        )
        
        self.patch_scorer = nn.Sequential(
            nn.Linear(self.patch_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def encode_rgb(self, x):
        features = self.rgb_encoder(x)
        return features
    
    def encode_point(self, x):
        xyz = x[:,0:3,:].permute(0, 2, 1)
        batch_size, _, _ = x.size()
        
        x = F.relu(self.bn1(self.conv1(x)))
        x_skip = F.relu(self.bn2(self.conv2(x)))

        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, neighbor=32, xyz=xyz, feature=x)
        feature_0 = self.gather_local_0(new_feature)

        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, neighbor=32, xyz=new_xyz, feature=feature_0)
        feature_1 = self.gather_local_1(new_feature)

        feature_1 = torch.cat((feature_1, x_skip), dim=1)
        features = self.conv_fuse1(feature_1)
        
        x = self.final_conv(features)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        return x

    def extract_rgb_features(self, rgb_data):
        x = rgb_data
        x = self.rgb_encoder.conv1(x)
        x = self.rgb_encoder.bn1(x)
        x = self.rgb_encoder.relu(x)
        x = self.rgb_encoder.maxpool(x)
        x = self.rgb_encoder.layer1(x)
        return x
    
    def QSS(self, rgb_data, pixel_coords, temperature=1.0):
        feature_maps = self.extract_rgb_features(rgb_data)
        batch_size = feature_maps.shape[0]

        normalized_coords = normalize_pixel_coordinates(pixel_coords.cpu().numpy(), original_size=1080, target_size=56)
        normalized_coords = torch.from_numpy(normalized_coords).to(pixel_coords.device)

        cropped_patches = crop_feature_maps_at_coordinates(
            feature_maps, normalized_coords, crop_size=self.crop_size
        )
        
        patch_features = []
        for i in range(9):
            patch = cropped_patches[:, i]
            patch_feat = self.patch_conv(patch)
            patch_features.append(patch_feat)
        
        patch_features = torch.stack(patch_features, dim=1)
        
        attn_output, _ = self.patch_attention(
            patch_features, patch_features, patch_features
        )
        
        patch_features = self.patch_norm1(patch_features + attn_output)
        
        ffn_output = self.patch_ffn(patch_features)
        
        patch_features = self.patch_norm2(patch_features + ffn_output)
        
        patch_scores = self.patch_scorer(patch_features).squeeze(-1)
        
        patch_logits = patch_scores / temperature
        patch_probs = F.softmax(patch_logits, dim=1)
        
        if self.training:
            from torch.distributions import Categorical
            dist = Categorical(probs=patch_probs)
            selected_idx = dist.sample()
        else:
            selected_idx = torch.argmax(patch_logits, dim=1)
        
        return selected_idx, patch_scores, patch_probs, patch_features
    
    def compute_policy_loss(self, patch_logits, selected_idx, reward):
        dist = Categorical(logits=patch_logits)  
        log_prob = dist.log_prob(selected_idx)

        policy_loss = -log_prob * reward.detach()
        
        return policy_loss.mean()
    
    def compute_reward(self, prediction, target):
        absolute_error = torch.abs(prediction - target).squeeze(1)
        reward = torch.exp(-(absolute_error) * 100 / 15.0)
        
        return reward
    
    def forward(self, rgb_data, point_data, is_random=True, pixel_coords=None, temperature=1.0):
        rgb_feat = self.encode_rgb(rgb_data)
        if is_random:
            point_feat = self.encode_point(point_data)
        else:
            if pixel_coords is not None:
                selected_patch_idx, patch_logits, patch_probs, patch_features = self.QSS(
                    rgb_data, pixel_coords, temperature=temperature
                )
                
                batch_size = point_data.shape[0]
                selected_point_data = point_data[torch.arange(batch_size), selected_patch_idx]
                point_feat = self.encode_point(selected_point_data)
            
            combined = torch.cat([rgb_feat, point_feat], dim=1)
            fused = self.fusion_layer(combined)
            
            return {
                'fused_features': fused,
                'selected_patch_idx': selected_patch_idx,
                'patch_logits': patch_logits,
                'patch_probs': patch_probs,
                'patch_features': patch_features
            }
        
        combined = torch.cat([rgb_feat, point_feat], dim=1)
        fused = self.fusion_layer(combined)
        return fused

class R3_PCQA(nn.Module):    
    def __init__(self, n_views=20, feature_dim=128, num_patches=9):
        super().__init__()
        self.n_views = n_views
        self.feature_dim = feature_dim
        self.num_patches = num_patches
        
        self.view_encoder = ViewEncoder(point_num=8192, feature_dim=feature_dim)
        
        self.num_heads = 3
        self.head_dim = (feature_dim + 1) // self.num_heads
        assert (feature_dim + 1) % self.num_heads == 0, "feature_dim + 1 must be divisible by num_heads"
        
        self.context_query = nn.Linear(feature_dim + 1, feature_dim + 1)
        self.context_key = nn.Linear(feature_dim + 1, feature_dim + 1)
        self.context_value = nn.Linear(feature_dim + 1, feature_dim + 1)
        self.output_projection = nn.Linear(feature_dim + 1, feature_dim + 1)
        self.attention_dropout = nn.Dropout(0.1)
        self.projection_dropout = nn.Dropout(0.1)
        self.context_norm = nn.LayerNorm(feature_dim + 1)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim + 1, (feature_dim + 1) * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear((feature_dim + 1) * 4, feature_dim + 1),
            nn.Dropout(0.1)
        )
        self.ff_norm = nn.LayerNorm(feature_dim + 1)
        
        self.view_regressor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.final_regressor = nn.Sequential(
            nn.Linear(2 * (feature_dim + 1), 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def global_context_attention(self, fused_features, view_predictions):
        feature_with_vp = torch.cat([view_predictions, fused_features], dim=-1)
        global_context = feature_with_vp.mean(dim=1, keepdim=True)
        
        batch_size, seq_len, embed_dim = feature_with_vp.shape
        _, context_len, _ = global_context.shape
        
        q = self.context_query(global_context)
        k = self.context_key(feature_with_vp)
        v = self.context_value(feature_with_vp)
        
        q = q.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        attended_output = torch.matmul(attention_weights, v)
        
        attended_output = attended_output.transpose(1, 2).contiguous().view(
            batch_size, context_len, embed_dim
        )
        
        attended_output = self.output_projection(attended_output)
        attended_output = self.projection_dropout(attended_output)
        
        attended_output = self.context_norm(attended_output + global_context)
        
        ff_output = self.feed_forward(attended_output)
        
        ff_output = self.ff_norm(attended_output + ff_output)
        
        combined_output = torch.cat([ff_output, global_context], dim=-1)
        
        attention_weights_avg = attention_weights.mean(dim=1)
        
        return combined_output.squeeze(1), attention_weights_avg.squeeze(1)
    
    def forward(self, data_dict, pixel_coords=None, is_random=True, temperature=1.0):
        rgb_views = data_dict['rgb']
        point_views = data_dict['point']
        
        batch_size = rgb_views.shape[0]
        
        fused_features = []
        view_predictions = []
        
        all_selected_patch_idx = []
        all_patch_logits = []
        all_patch_probs = []
        all_patch_features = []
        
        for i in range(self.n_views):
            rgb_view = rgb_views[:, i]
            point_view = point_views[:, i]
            
            if is_random:
                fused = self.view_encoder(rgb_view, point_view, is_random=True)
                fused_features.append(fused)
            else:
                view_pixel_coords = pixel_coords[:, i]
                view_output = self.view_encoder(
                    rgb_view, point_view, is_random=False, 
                    pixel_coords=view_pixel_coords, temperature=temperature
                )
                
                fused = view_output['fused_features']
                all_selected_patch_idx.append(view_output['selected_patch_idx'])
                all_patch_logits.append(view_output['patch_logits'])
                all_patch_probs.append(view_output['patch_probs'])
                all_patch_features.append(view_output['patch_features'])
                
                fused_features.append(fused)
            
            view_pred = self.view_regressor(fused)
            view_predictions.append(view_pred)
        
        fused_features = torch.stack(fused_features, dim=1)
        view_predictions = torch.stack(view_predictions, dim=1)
        
        global_context, attention_weights = self.global_context_attention(fused_features, view_predictions)
        
        final_prediction = self.final_regressor(global_context)
        
        result = {
            'final_prediction': final_prediction,
            'view_predictions': view_predictions,
            'attention_weights': attention_weights
        }
        
        if not is_random:
            result['selected_patch_idx'] = torch.stack(all_selected_patch_idx, dim=1)
            result['patch_logits'] = torch.stack(all_patch_logits, dim=1)
            result['patch_probs'] = torch.stack(all_patch_probs, dim=1)
            result['patch_features'] = torch.stack(all_patch_features, dim=1)
        
        return result
    
    def compute_contextual_bandit_loss(self, outputs, target):
        if 'selected_patch_idx' not in outputs:
            raise ValueError("Contextual Bandit information is missing. Please run with is_random=False.")
        
        reward = self.view_encoder.compute_reward(
            outputs['final_prediction'], 
            target
        )
        
        attention_weights = outputs['attention_weights']
        selected_patch_idx = outputs['selected_patch_idx']
        patch_logits = outputs['patch_logits']
        
        batch_size, num_views = selected_patch_idx.shape
        
        view_rewards = reward.unsqueeze(1) * attention_weights
        
        total_policy_loss = 0
        
        for view_idx in range(num_views):
            view_selected_idx = selected_patch_idx[:, view_idx]
            view_patch_logits = patch_logits[:, view_idx, :]
            view_reward = view_rewards[:, view_idx]
            
            view_policy_loss = self.view_encoder.compute_policy_loss(
                view_patch_logits,
                view_selected_idx,
                view_reward
            )
            
            total_policy_loss += view_policy_loss
        
        avg_policy_loss = total_policy_loss / num_views
        
        return avg_policy_loss

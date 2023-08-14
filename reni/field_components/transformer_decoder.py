import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, direction_input_dim, conditioning_input_dim, latent_dim, num_heads):
        super().__init__()
        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(direction_input_dim, latent_dim)
        self.key = nn.Linear(conditioning_input_dim, latent_dim)
        self.value = nn.Linear(conditioning_input_dim, latent_dim)
        self.fc_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention = torch.einsum("bnqk,bnkh->bnqh", [Q, K.transpose(-2, -1)]) * self.scale
        attention = torch.softmax(attention, dim=-1)

        out = torch.einsum("bnqh,bnhv->bnqv", [attention, V])
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        out = self.fc_out(out).squeeze(1)
        return out

class AttentionLayer(nn.Module):
    def __init__(self, direction_input_dim, conditioning_input_dim, latent_dim, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(direction_input_dim, conditioning_input_dim, latent_dim, num_heads)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, directional_input, conditioning_input):
        attn_output = self.mha(directional_input, conditioning_input, conditioning_input)
        out1 = self.norm1(attn_output + directional_input)
        fc_output = self.fc(out1)
        out2 = self.norm2(fc_output + out1)
        return out2
    

class Decoder(nn.Module):
    def __init__(self, in_dim, conditioning_input_dim, hidden_features, num_heads, num_layers):
        super().__init__()
        self.residual_projection = nn.Linear(in_dim, hidden_features)  # projection for residual connection
        self.layers = nn.ModuleList(
            [AttentionLayer(hidden_features, conditioning_input_dim, hidden_features, num_heads)
             for i in range(num_layers)]
        )  
        self.fc = nn.Linear(hidden_features, 3)  # 3 for RGB

    def forward(self, x, conditioning_input):
        x = self.residual_projection(x)
        for layer in self.layers:
            x = layer(x, conditioning_input)
        return self.fc(x)
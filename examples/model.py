import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.nn import functional as F
import math

class SequencePositionalEncoding(nn.Module):
    """Positional encoding for orderbook sequences"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :].to(x.device)

class ActorCritic(nn.Module):
    """
    Actor-Critic network with LSTM + Transformer for sequence processing
    Handles stacked orderbook frames
    """
    
    def __init__(self, 
                observation_shape: Tuple[int, int],  # (sequence_length, features)
                lstm_hidden_size: int = 128, 
                transformer_heads: int = 4,
                transformer_layers: int = 2,
                action_dim: int = 7,
                dropout: float = 0.1,
                use_positional_encoding: bool = True):
        super().__init__()
        
        self.sequence_length, self.input_size = observation_shape
        self.lstm_hidden_size = lstm_hidden_size
        self.action_dim = action_dim
        self.use_positional_encoding = use_positional_encoding
        
        print(f"ActorCritic initialized with:")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Input size per frame: {self.input_size}")
        print(f"  LSTM hidden size: {lstm_hidden_size}")
        print(f"  Transformer heads: {transformer_heads}")
        print(f"  Transformer layers: {transformer_layers}")
        
        # Input projection for each frame
        self.input_proj = nn.Linear(self.input_size, lstm_hidden_size)
        
        # LSTM for processing sequences
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size, 
            num_layers=1,
            batch_first=True,
            dropout=dropout if transformer_layers > 1 else 0
        )
        
        # Positional encoding for transformer
        if self.use_positional_encoding:
            self.pos_encoding = SequencePositionalEncoding(lstm_hidden_size)
        
        # Transformer layers for capturing temporal patterns
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden_size,
            nhead=transformer_heads,
            dim_feedforward=lstm_hidden_size * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=transformer_layers
        )
        
        # Attention mechanism for sequence aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size,
            num_heads=transformer_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output heads
        self.actor_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size // 2, action_dim)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for sequence of orderbook frames
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
            hidden: LSTM hidden state tuple (h, c)
            
        Returns:
            action_logits: (batch_size, action_dim) 
            value: (batch_size, 1)
            new_hidden: Updated LSTM hidden state
        """
        batch_size, seq_len, features = x.shape
        
        # Ensure input is the right shape
        assert seq_len == self.sequence_length, f"Expected sequence length {self.sequence_length}, got {seq_len}"
        assert features == self.input_size, f"Expected {self.input_size} features, got {features}"
        
        # Project each frame
        # x shape: (batch_size, seq_len, features) -> (batch_size, seq_len, lstm_hidden_size)
        x_proj = F.relu(self.input_proj(x))
        
        # LSTM processing - handles the sequence
        lstm_out, new_hidden = self.lstm(x_proj, hidden)
        # lstm_out shape: (batch_size, seq_len, lstm_hidden_size)
        
        # Apply positional encoding if enabled
        if self.use_positional_encoding:
            lstm_out = self.pos_encoding(lstm_out)
        
        # Transformer processing for temporal dependencies
        transformer_out = self.transformer(lstm_out)
        # transformer_out shape: (batch_size, seq_len, lstm_hidden_size)
        
        # Aggregate sequence using attention
        # Use the last timestep as query, attending to all timesteps
        query = transformer_out[:, -1:, :]  # (batch_size, 1, lstm_hidden_size)
        attn_out, attn_weights = self.attention(query, transformer_out, transformer_out)
        
        # Use the attended output for final predictions
        features = attn_out.squeeze(1)  # (batch_size, lstm_hidden_size)
        
        # Generate outputs
        action_logits = self.actor_head(features)  # (batch_size, action_dim)
        value = self.critic_head(features)  # (batch_size, 1)
        
        return action_logits, value, new_hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state"""
        h = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        return (h, c)
    
    def get_action_and_value(self, x: torch.Tensor, 
                        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                        action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action and value for PPO training
        
        Args:
            x: Observation tensor (batch_size, sequence_length, features)
            hidden: LSTM hidden state
            action: Optional action tensor for computing log prob
            
        Returns:
            action: Selected action
            log_prob: Log probability of action  
            value: State value
            entropy: Action entropy
        """
        action_logits, value, new_hidden = self.forward(x, hidden)
        
        # Create action distribution
        dist = torch.distributions.Categorical(logits=action_logits)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, value, entropy

model = ActorCritic(observation_shape=(3,83), lstm_hidden_size=128, transformer_heads=4, transformer_layers=2, action_dim=4, dropout=0.1, use_positional_encoding=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
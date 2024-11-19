import torch
import torch.nn as nn
import torchvision
import open_clip

from transformers import BertModel
from open_clip.transformer import text_global_pool


class VisualEncoder(nn.Module):
    """
    Encode images using pre-traineds.
    """
    def __init__(self, model: str, finetune: bool, embedding_dim: int):
        super().__init__()
        self.model = model

        if self.model == 'resnet':
            self.embedding_dim = embedding_dim
            resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            modules = list(resnet.children())[:-1]  # remove the last layer
            self.layers = nn.Sequential(*modules)
        elif self.model == 'clip':
            self.embedding_dim = embedding_dim
            clip , _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            self.layers = clip.visual

        if not finetune:
            # freeze the parameters
            for param in self.layers.parameters():
                param.requires_grad = False 


    def forward(self, x: torch.Tensor):
        if self.model == 'resnet':
            # (batch_size, embedding_dim = 512)
            x = self.layers(x)
            x = x.squeeze(dim=(2, 3))
        elif self.model == 'clip':
            # (batch_size, embedding_dim = 512)
            x = self.layers(x)
            x /= x.norm(dim=-1, keepdim=True)
        return x


class TextEncoder(nn.Module):
    """
    Encode texts using pre-trained models, obtaining both token-level and sentence-level embeddings.
    """
    def __init__(self, model: str, num_layers: int, embedding_dim: int, hidden_dim: int, dropout: int):
        super().__init__()
        self.model = model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if self.model == 'bert':
            self.embedding_dim = embedding_dim
            self.layers = BertModel.from_pretrained('bert-base-uncased')

            # freeze the parameters
            for param in self.layers.parameters():
                param.requires_grad = False

            self.gru = nn.GRU(
                embedding_dim, hidden_dim, num_layers=num_layers, 
                bidirectional=True, batch_first=True, dropout=0 if num_layers < 2 else dropout
            )
        elif self.model == 'clip':
            self.embedding_dim = embedding_dim
            clip , _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k') 
            
            # freeze the parameters
            for param in clip.parameters():
                param.requires_grad = False

            # extract token-level context features
            self.transformer = clip.transformer
            self.context_length = clip.context_length
            self.vocab_size = clip.vocab_size
            self.token_embedding = clip.token_embedding
            self.positional_embedding = clip.positional_embedding
            self.ln_final = clip.ln_final
            self.text_projection = clip.text_projection
            self.text_pool_type = clip.text_pool_type
            self.attn_mask = clip.attn_mask


    def forward(self, text: torch.Tensor, attention_masks: torch.Tensor):
        if self.model == 'bert':
            # (batch_size, seq_length, embedding_dim = 768)
            # using the last hidden state of BERT as the embedding
            outputs = self.layers(input_ids=text, attention_mask=attention_masks, output_attentions=False)
            x = outputs['last_hidden_state']

            # bidirectional GRU
            # (batch_size, seq_length, hidden_dim * 2), (num_layers * 2, batch_size, hidden_dim)
            output, h = self.gru(x)
            h1 = output[..., :self.hidden_dim]
            h2 = output[..., self.hidden_dim:]
            # assert(torch.equal(h1[:, -1, :], h[self.num_layers * 2 - 2]))
            # assert(torch.equal(h2[:, 0, :], h[self.num_layers * 2 - 1]))

            # token-level context feature
            # (batch_size, seq_length, hidden_dim)
            w = (h1 + h2) / 2
            w = w * attention_masks.unsqueeze(dim=-1) # masking the padding tokens

            # sentence-level context feature
            # (batch_size, hidden_dim)
            s = torch.sum(w, dim=1) / torch.sum(attention_masks, dim=1, keepdim=True) # masking the padding tokens 
        elif self.model == 'clip':
            cast_dtype = self.transformer.get_cast_dtype()
            self.attn_mask = self.attn_mask.to(text.device)

            # (batch_size, seq_length, 512)
            x = self.token_embedding(text).to(cast_dtype)
            x = x + self.positional_embedding.to(cast_dtype)
            x = self.transformer(x, attn_mask=self.attn_mask)
            x = self.ln_final(x)
            # (batch_size, 512), (batch_size, seq_length, 512)
            s, w = text_global_pool(x, text, self.text_pool_type)
            # (batch_size, embedding_dim = 512)
            s = s @ self.text_projection
            s /= s.norm(dim=-1, keepdim=True)
            # (batch_size, seq_length, embedding_dim = 512)
            w = w @ self.text_projection
            w /= w.norm(dim=-1, keepdim=True)
        return w, s


class Attention(nn.Module):
    """
    Attention mechanism without a value-projection matrix.
    """
    def __init__(self, q_dim: int, k_dim: int, embedding_dim: int, generator: torch.Generator):
        super().__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.embedding_dim = embedding_dim

        self.w_q = nn.Linear(self.q_dim ,self.embedding_dim, bias=False)
        self.w_k = nn.Linear(self.k_dim ,self.embedding_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = self.embedding_dim ** -0.5

        self.initialize_parameters(generator)


    def initialize_parameters(self, generator: torch.Generator):
        """
        Initialize the weights using Xavier uniform distribution.
        """
        nn.init.xavier_uniform_(self.w_q.weight, generator=generator)
        nn.init.xavier_uniform_(self.w_k.weight, generator=generator)


    def forward(self, q: torch.Tensor, k: torch.Tensor, attn_mask: torch.Tensor = None):
        # (batch_size, num_regions, embedding_dim)
        q = self.w_q(q)
        # (batch_size, seq_length, embedding_dim)
        k = self.w_k(k)
        
        # adjust attention mask to ensure softmax scores are zero for masked regions.
        if attn_mask is not None:
            num_regions = q.shape[1]
            attn_mask = attn_mask.unsqueeze(1).expand(-1, num_regions, -1)
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(~attn_mask.bool(), float("-inf"))
            attn_mask = new_attn_mask
        
        # (batch_size, num_regions, seq_length)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn += attn_mask
        attn = self.softmax(attn)
        # (batch_size, num_regions, embedding_dim)
        x = torch.matmul(attn, k)
        return x


class ITIN(nn.Module):
    """
    A multimodal sentiment analysis model that integrates image and text data.
    This model enhances the interaction between visual and textual inputs during 
    the fusion process by aligning image regions with corresponding textual elements.
    
    Components of the model:
    - Visual Encoder
    - Text Encoder
    - Object Detection Module
    - Cross-Modal Alignment Module
    - Cross-Modal Gating Module
    - Multimodal Sentiment Classifier
    """
    def __init__(self, cfg, generator: torch.Generator):
        super().__init__()
        self.v_base = cfg.model.visual.train_baseline
        self.t_base = cfg.model.text.train_baseline
        self.cm_base = cfg.model.cross_modal_baseline
        self.c_base = cfg.model.context_baseline
        self.wo_alignment = cfg.model.without_alignment
        self.wo_gating = cfg.model.without_gating
        self.hidden_dim1 = cfg.model.hidden_dim1
        self.hidden_dim2 = cfg.model.hidden_dim2
        self.embedding_dim = cfg.model.embedding_dim
        self.dropout = cfg.model.dropout
        self._lambda = cfg.model._lambda
        self.num_attns = cfg.model.num_attns
        self.num_classes = cfg.model.num_classes
        
        # context information extraction
        self.visual_encoder = VisualEncoder(
            model=cfg.model.visual.model, 
            finetune=cfg.model.visual.finetune,
            embedding_dim=cfg.model.visual.embedding_dim
        )
        self.text_encoder = TextEncoder(
            model=cfg.model.text.model,
            num_layers=cfg.model.text.num_layers, 
            embedding_dim=cfg.model.text.embedding_dim,
            hidden_dim=self.hidden_dim1,
            dropout=cfg.model.text.dropout
        )

        # linear project to a d-dimentional regional feature
        self.fc_region = nn.Linear(cfg.model.region.embedding_dim, self.hidden_dim1)

        if self.v_base:
            self.fc_sentiment = nn.Linear(self.visual_encoder.embedding_dim, self.num_classes)
        if self.t_base:
            self.fc_sentiment = nn.Linear(self.hidden_dim1, self.num_classes)
        if self.c_base:
            self.mlp_sentiment_visual = nn.Sequential(
                nn.Linear(self.visual_encoder.embedding_dim, self.hidden_dim2),
                nn.ReLU()
            )
            self.mlp_sentiment_text = nn.Sequential(
                nn.Linear(self.hidden_dim1, self.hidden_dim2),
                nn.ReLU()
            )
            self.fc_sentiment = nn.Linear(self.hidden_dim2, self.num_classes)
        
        # cross-modal alignment module
        self.cross_attn = Attention(self.hidden_dim1, self.hidden_dim1, self.embedding_dim, generator)
        self.self_attn = Attention(self.hidden_dim1, self.hidden_dim1, self.embedding_dim, generator)
        
        # cross-modal gating module
        self.sigmoid = nn.Sigmoid()
        self.fc_gating = nn.Linear(self.hidden_dim1 * 2, self.hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout)

        # use region-feature dependent weights to obtain a fused alignment feature
        self.mlp_gating = nn.Sequential(
            nn.Linear(self.hidden_dim1, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, 4)
        )
        self.softmax = nn.Softmax(dim=-1)

        if self.cm_base:
            self.fc_sentiment = nn.Linear(self.hidden_dim1, self.num_classes)

        # multimodal sentiment classification
        if not (self.v_base or self.t_base or self.cm_base or self.c_base):
            self.mlp_sentiment_visual = nn.Sequential(
                nn.Linear(self.visual_encoder.embedding_dim + self.hidden_dim1, self.hidden_dim2),
                nn.ReLU()
            )
            self.mlp_sentiment_text = nn.Sequential(
                nn.Linear(self.hidden_dim1 * 2, self.hidden_dim2),
                nn.ReLU()
            )
            self.fc_sentiment = nn.Linear(self.hidden_dim2, self.num_classes)

        self.initialize_parameters(generator)


    def initialize_parameters(self, generator: torch.Generator):
        """
        Initialize the weights using Xavier uniform distribution.
        """
        nn.init.xavier_uniform_(self.fc_region.weight, generator=generator)
        nn.init.zeros_(self.fc_region.bias)

        nn.init.xavier_uniform_(self.fc_gating.weight, generator=generator)
        nn.init.zeros_(self.fc_gating.bias)
        nn.init.xavier_uniform_(self.mlp_gating[0].weight, generator=generator)
        nn.init.zeros_(self.mlp_gating[0].bias)
        nn.init.xavier_uniform_(self.mlp_gating[2].weight, generator=generator)
        nn.init.zeros_(self.mlp_gating[2].bias)

        if not (self.v_base or self.t_base or self.cm_base):
            nn.init.xavier_uniform_(self.mlp_sentiment_visual[0].weight, generator=generator)
            nn.init.zeros_(self.mlp_sentiment_visual[0].bias)
            nn.init.xavier_uniform_(self.mlp_sentiment_text[0].weight, generator=generator)
            nn.init.zeros_(self.mlp_sentiment_text[0].bias)

        nn.init.xavier_uniform_(self.fc_sentiment.weight, generator=generator)
        nn.init.zeros_(self.fc_sentiment.bias)


    def l2_norm(self, x: torch.Tensor, dim: int = -1, eps: int = 1e-8):
        """
        Perform L2 normalization along the columns of x.
        """
        norm = torch.pow(x, 2).sum(dim=dim, keepdim=True).sqrt() + eps
        x = torch.div(x, norm)
        return x


    def forward(
        self, images: torch.Tensor, texts: torch.Tensor, attention_masks: torch.Tensor, 
        region_features: torch.Tensor
    ):
        # context information extraction
        # (batch_size, embedding_dim) 
        v = self.visual_encoder(images)
        if self.v_base:
            # (batch_size, num_classes) 
            x = self.fc_sentiment(v)
            return x
        
        # (batch_size, seq_length, hidden_dim1), (batch_size, hidden_dim1)
        w, s = self.text_encoder(texts, attention_masks)
        if self.t_base:
            # (batch_size, num_classes) 
            x = self.fc_sentiment(s)
            return x

        if self.c_base:
            # (batch_size, hidden_dim2)
            f1 = self.mlp_sentiment_visual(v)
            # (batch_size, hidden_dim2)
            f2 = self.mlp_sentiment_text(s)
            # (batch_size, hidden_dim2)
            f = self._lambda * f1 + (1 - self._lambda) * f2
            # (batch_size, num_classes)
            x = self.fc_sentiment(f)
            return x
        
        # (batch_size, num_regions, hidden_dim1)
        r = self.fc_region(region_features)
        r = self.l2_norm(r, dim=-1)
        
        # cross-modal alignment module
        if not self.wo_alignment:
            # (batch_size, num_regions, hidden_dim1)
            u = self.cross_attn(r, w, attention_masks)
            # (batch_size, num_regions, hidden_dim1)
            u = self.self_attn(u, u)
            u = self.l2_norm(u, dim=-1)
        else:
            # (batch_size, num_regions, hidden_dim1)
            u = r
        
        # cross-modal gating module
        if not self.wo_gating:
            # (batch_size, num_regions)
            g = self.sigmoid(torch.sum(r * u, dim=-1))
            # (batch_size, num_regions, hidden_dim1 * 2)
            c = g.unsqueeze(dim=-1) * torch.cat((r, u), dim=-1)
            # (batch_size, num_regions, hidden_dim1)
            o = self.dropout(self.relu1(self.fc_gating(c)))
            z = o + r
            
            # (batch_size, num_regions, num_attns)
            a = self.softmax(self.mlp_gating(z))
            # (btach_size, num_attns, hidden_dim1)
            c = torch.matmul(a.transpose(-2, -1), z)
            # (batch_size, hidden_dim1)
            c = c.mean(dim=-2)
            c = self.l2_norm(c, dim=-1)
        else:
            # (batch_size, hidden_dim1)
            c = u.mean(dim=-2)
            c = self.l2_norm(c, dim=-1)

        if self.cm_base:
            # (batch_size, num_classes) 
            x = self.fc_sentiment(c)
            return x
        
        # multimodal sentiment classification
        # (batch_size, hidden_dim2)
        f1 = self.mlp_sentiment_visual(torch.cat((v, c), dim=-1))
        # (batch_size, hidden_dim2)
        f2 = self.mlp_sentiment_text(torch.cat((s, c), dim=-1))
        # (batch_size, hidden_dim2)
        f = self._lambda * f1 + (1 - self._lambda) * f2
        # (batch_size, num_classes)
        x = self.fc_sentiment(f)
        return x
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    PreTrainedModel,
    WhisperConfig,
)
from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
from dataclasses import dataclass
from typing import Optional
import json
import math
from whistress.model.modules.net_utils import MeanPooling
from losses import (
    compute_adaptive_weighted_loss,
    compute_cross_granularity_conditional_ranking_loss,
    compute_word_level_mil_loss,
)


@dataclass
class CustomModelOutput(BaseModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    head_preds: torch.FloatTensor = None
    labels_head: Optional[torch.FloatTensor] = None
    whisper_logits: torch.FloatTensor = None
    preds: Optional[torch.Tensor] = None

@dataclass
class CustomPhnModelOutput(BaseModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_main: Optional[torch.FloatTensor] = None
    loss_wsd: Optional[torch.FloatTensor] = None
    loss_wsl: Optional[torch.FloatTensor] = None
    loss_rank: Optional[torch.FloatTensor] = None
    loss_mil: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    phone_stress_logits: torch.FloatTensor = None
    head_preds: torch.FloatTensor = None
    labels_head: Optional[torch.FloatTensor] = None
    whisper_logits: torch.FloatTensor = None
    preds: Optional[torch.Tensor] = None
    phone_stress_preds: Optional[torch.Tensor] = None

# Define a new head (e.g., a classification layer)
class LinearHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearHead, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class FCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCNN, self).__init__()
        hidden_dim = 2 * input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PosBias(nn.Module):
    SUPPORTED_MODES = ("static", "scalar_gated")

    def __init__(self, d_model, pos_bias_config):
        super().__init__()
        self.mode = pos_bias_config["mode"]
        if self.mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported POS bias mode: {self.mode}")

        self.pos_embed_dim = pos_bias_config["embedding_dim"]
        self.pos_embed = nn.Embedding(
            num_embeddings=pos_bias_config["num_pos"] + 1,
            embedding_dim=self.pos_embed_dim,
            padding_idx=0,
        )
        self.pos_proj = nn.Linear(self.pos_embed_dim, d_model, bias=False)
        self.pos_dropout = nn.Dropout(pos_bias_config["dropout"])

        self.residual_scale = None
        self.pos_gate_query = None
        self.pos_gate_key = None
        self.pos_gate_bias = None
        self.pos_value_norm = None

        if self.mode == "static":
            self.residual_scale = nn.Parameter(
                torch.tensor(float(pos_bias_config["residual_scale_init"]))
            )
        else:
            self.gate_dim = int(
                pos_bias_config.get("gate_dim", self.pos_embed_dim)
            )
            if self.gate_dim <= 0:
                raise ValueError("gate_dim must be positive")

            gate_init = float(pos_bias_config.get("gate_init", 0.01))
            if not 0.0 < gate_init < 1.0:
                raise ValueError("gate_init must be strictly between 0 and 1")

            self.pos_gate_query = nn.Linear(
                self.pos_embed_dim, self.gate_dim, bias=False
            )
            self.pos_gate_key = nn.Linear(d_model, self.gate_dim, bias=False)
            self.pos_gate_bias = nn.Parameter(
                torch.tensor(math.log(gate_init / (1.0 - gate_init)))
            )
            nn.init.zeros_(self.pos_gate_query.weight)

            if pos_bias_config.get("normalize_pos_value", True):
                self.pos_value_norm = nn.LayerNorm(
                    d_model, elementwise_affine=False
                )

    def forward(self, hidden_states, token_pos_ids=None):
        if token_pos_ids is None:
            return hidden_states

        valid_pos_mask = (token_pos_ids >= 0).unsqueeze(-1)
        pos_input_ids = token_pos_ids + 1
        pos_embed_low = self.pos_embed(pos_input_ids)
        pos_bias = self.pos_proj(pos_embed_low)
        pos_bias = self.pos_dropout(pos_bias)

        if self.mode == "scalar_gated":
            query = self.pos_gate_query(pos_embed_low)
            key = self.pos_gate_key(hidden_states)
            gate_logits = (
                (query * key).sum(dim=-1, keepdim=True)
                / math.sqrt(self.gate_dim)
                + self.pos_gate_bias
            )
            gate = torch.sigmoid(gate_logits)
            if self.pos_value_norm is not None:
                pos_bias = self.pos_value_norm(pos_bias)
            pos_bias = gate * pos_bias

        pos_bias = pos_bias * valid_pos_mask
        if self.mode == "static":
            pos_bias = self.residual_scale * pos_bias
        return hidden_states + pos_bias


class WhiStress(PreTrainedModel):

    config_class = WhisperConfig
    model_input_names = ["input_features", "labels_head", "whisper_labels"]

    def __init__(
        self,
        config: WhisperConfig,
        layer_for_head: Optional[int] = None,
        whisper_backbone_name="openai/whisper-small.en",
        class_weights = [1.0, 2.33],
        loss_lambdas=None
    ):
        super().__init__(config)
        self.whisper_backbone_name = whisper_backbone_name
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            self.whisper_backbone_name,
        ).eval()
        self.processor = WhisperProcessor.from_pretrained(self.whisper_backbone_name)

        input_dim = self.whisper_model.config.d_model  # Model's hidden size
        output_dim = 2  # Number of classes or output features for the new head

        config = self.whisper_model.config
        # add additional decoder block using the existing Whisper config
        self.additional_decoder_block = WhisperDecoderLayer(config)
        self.classifier = FCNN(input_dim, output_dim)
        # add weighted loss for CE
        class_weights = torch.tensor(class_weights)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
        self.layer_for_head = -1 if layer_for_head is None else layer_for_head

        if loss_lambdas:
            self.lambda_ssd = loss_lambdas["lambda_ssd"]
            self.lambda_wsl = loss_lambdas["lambda_wsl"]
        else:
            self.lambda_ssd = 1
            self.lambda_wsl = -1

    def to(self, device: str = ("cuda" if torch.cuda.is_available() else "cpu")):
        self.whisper_model.to(device)
        self.additional_decoder_block.to(device)
        self.classifier.to(device)
        super().to(device)
        return self

    def load_model(self, save_dir=None):
        # load only the classifier and extra decoder layer (saved locally)
        if save_dir is not None:
            print('loading model from:', save_dir)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.classifier.load_state_dict(
                torch.load(
                    os.path.join(save_dir, "classifier.pt"),
                    weights_only=False,
                    map_location=torch.device(device),
                )
            )
            self.additional_decoder_block.load_state_dict(
                torch.load(
                    os.path.join(save_dir, "additional_decoder_block.pt"),
                    weights_only=False,
                    map_location=torch.device(device),
                )
            )
            # read and load the layer_for_head.json
            # the json format is {"layer_for_head": 9}
            with open(os.path.join(save_dir, "metadata.json"), "r") as f:
                metadata = json.load(f)
                self.layer_for_head = metadata["layer_for_head"]

    def train(self, mode: Optional[bool] = True):
        # freeze whisper and train classifier
        super().train(mode)
        # freeze whisper and train task heads only
        self.whisper_model.eval()
        # mark whisper model requires grad false
        for param in self.whisper_model.parameters():
            param.requires_grad = False
        for param in self.additional_decoder_block.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        self.additional_decoder_block.train(mode)
        self.classifier.train(mode)
        return self

    def eval(self):
        super().eval()
        self.whisper_model.eval()
        self.additional_decoder_block.eval()
        self.classifier.eval()
        return self

    def forward(
        self,
        input_features,
        attention_mask=None,
        decoder_input_ids=None,
        labels_head=None,
        whisper_labels=None,
        phone_ids=None,
        phone_labels_head=None,
        token_pos_ids=None,
        word_ids=None
    ):  
        device = input_features.device
        self.whisper_model.eval()
         
        # pass the inputs through the model
        backbone_outputs = self.whisper_model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            labels=whisper_labels,
        )

        # Extract the hidden states of the last layer of the decoder
        decoder_last_layer_hidden_states = backbone_outputs.decoder_hidden_states[
            self.layer_for_head
        ].to(device)

        # Extract the hidden states of the layer of the encoder who encapsulates best the prosodic features
        layer_for_head_hidden_states = backbone_outputs.encoder_hidden_states[
            self.layer_for_head
        ].to(device)
        # Pass the decoder last hidden layers through the new head (decoder_block + lin cls)

        additional_decoder_block_outputs = self.additional_decoder_block(
            hidden_states=decoder_last_layer_hidden_states,
            encoder_hidden_states=layer_for_head_hidden_states,
        )
        head_logits = self.classifier(additional_decoder_block_outputs[0].to(device))

        # calculate softmax
        head_probs = F.softmax(head_logits, dim=-1)
        preds = head_probs.argmax(dim=-1).to(device)
        # Calculate custom loss if labels are provided
        # sentence stress detection
        loss = None
        loss_main = None
        if labels_head is not None:
            preds = torch.where(
                torch.isin(
                    labels_head, torch.tensor(list([-100])).to(device)  # 50257, 50362,
                ),
                torch.tensor(-100),
                preds,
            )
            # CrossEntropyLoss for the custom head
            loss_main = self.loss_fct(
                head_logits.reshape(-1, head_logits.size(-1)), labels_head.reshape(-1)
            )
            loss = self.lambda_ssd * loss_main
        
        # word stress loss
        loss_wsl = None
        if word_ids is not None and labels_head is not None and self.lambda_wsl > 0.0:
            loss_wsl = compute_adaptive_weighted_loss(head_logits, labels_head, word_ids)
            loss += self.lambda_wsl * loss_wsl

        return CustomPhnModelOutput(
            logits=head_logits,
            labels_head=labels_head,
            whisper_logits=backbone_outputs.logits,
            loss=loss,
            loss_main=loss_main,
            loss_wsl=loss_wsl,
            preds=preds,
        )

    def generate(
        self,
        input_features,
        max_length=128,
        labels_head=None,
        whisper_labels=None,
        **generate_kwargs,
    ):
        """
        Generate both the Whisper output and custom head output sequences in alignment.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Generate the Whisper output sequence
        whisper_outputs = self.whisper_model.generate(
            input_features=input_features,
            max_length=max_length,
            labels=whisper_labels,
            do_sample=False,
            **generate_kwargs,
        )

        # pass the inputs through the model
        backbone_outputs = self.whisper_model(
            input_features=input_features,
            decoder_input_ids=whisper_outputs,
            output_hidden_states=True,
        )

        # Extract the hidden states of the last layer of the decoder
        decoder_last_layer_hidden_states = backbone_outputs.decoder_hidden_states[
            self.layer_for_head
        ].to(device)

        # Extract the hidden states of the last layer of the encoder
        layer_for_head_hidden_states = backbone_outputs.encoder_hidden_states[
            self.layer_for_head
        ].to(device)
        # Pass the decoder last hidden layers through the new head (decoder_block + lin cls)

        additional_decoder_block_outputs = self.additional_decoder_block(
            hidden_states=decoder_last_layer_hidden_states,
            encoder_hidden_states=layer_for_head_hidden_states,
        )
        head_logits = self.classifier(additional_decoder_block_outputs[0].to(device))
        # calculate softmax
        head_probs = F.softmax(head_logits, dim=-1)
        preds = head_probs.argmax(dim=-1).to(device)
        preds = torch.where(
            torch.isin(
                whisper_outputs, torch.tensor(list([50256])).to(device)  # 50257, 50362,
            ),
            torch.tensor(-100),
            preds,
        )
        return preds

    def generate_dual(
        self,
        input_features,
        attention_mask=None,
        max_length=200,
        labels_head=None,
        whisper_labels=None,
        **generate_kwargs,
    ):
        """
        Generate both the Whisper output and custom head output sequences in alignment.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Generate the Whisper output sequence
        whisper_outputs = self.whisper_model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            max_length=max_length,
            labels=whisper_labels,
            return_dict_in_generate=True,
            **generate_kwargs,
        )

        # pass the inputs through the model
        backbone_outputs = self.whisper_model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=whisper_outputs.sequences,
            output_hidden_states=True,
        )

        # Extract the hidden states of the last layer of the decoder
        decoder_last_layer_hidden_states = backbone_outputs.decoder_hidden_states[
            self.layer_for_head
        ].to(device)

        # Extract the hidden states of the last layer of the encoder
        layer_for_head_hidden_states = backbone_outputs.encoder_hidden_states[
            self.layer_for_head
        ].to(device)
        # Pass the decoder last hidden layers through the new head (decoder_block + lin cls)

        additional_decoder_block_outputs = self.additional_decoder_block(
            hidden_states=decoder_last_layer_hidden_states,
            encoder_hidden_states=layer_for_head_hidden_states,
        )
        head_logits = self.classifier(additional_decoder_block_outputs[0].to(device))
        head_probs = F.softmax(head_logits, dim=-1)
        preds = head_probs.argmax(dim=-1).to(device)
        preds = torch.where(
            torch.isin(
                whisper_outputs.sequences, torch.tensor(list([50256])).to(device)  # 50257, 50362,
            ),
            torch.tensor(-100),
            preds,
        )

        return CustomPhnModelOutput(
            logits=head_logits,
            head_preds=preds,
            whisper_logits=whisper_outputs.logits,
            preds=whisper_outputs.sequences,
            
        )

    def __str__(self):
        return "WhiStress"


class WhiStressPos(WhiStress):
    def __init__(self, *args, pos_bias_config=None,
                 freeze_pretrained_heads=False, **kwargs):
        if pos_bias_config is None:
            raise ValueError("pos_bias_config is required for WhiStressPos")
        super().__init__(*args, **kwargs)
        self.pos_bias = PosBias(self.whisper_model.config.d_model, pos_bias_config)
        self.freeze_pretrained_heads = freeze_pretrained_heads

    def _freeze_pretrained_heads(self):
        for module in [self.additional_decoder_block]:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_features,
        attention_mask=None,
        decoder_input_ids=None,
        labels_head=None,
        whisper_labels=None,
        phone_ids=None,
        phone_labels_head=None,
        token_pos_ids=None,
        word_ids=None
    ):
        device = input_features.device
        self.whisper_model.eval()

        # pass the inputs through the model
        backbone_outputs = self.whisper_model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            labels=whisper_labels,
        )

        # Extract the hidden states of the last layer of the decoder
        decoder_last_layer_hidden_states = backbone_outputs.decoder_hidden_states[
            self.layer_for_head
        ].to(device)
        # Extract the hidden states of the layer of the encoder who encapsulates best the prosodic features
        layer_for_head_hidden_states = backbone_outputs.encoder_hidden_states[
            self.layer_for_head
        ].to(device)
        # Pass the decoder last hidden layers through the new head (decoder_block + lin cls)

        additional_decoder_block_outputs = self.additional_decoder_block(
            hidden_states=decoder_last_layer_hidden_states,
            encoder_hidden_states=layer_for_head_hidden_states,
        )
        ssd_hidden_states = additional_decoder_block_outputs[0].to(device)
        ssd_hidden_states = self.pos_bias(ssd_hidden_states, token_pos_ids)
        head_logits = self.classifier(ssd_hidden_states)

        # calculate softmax
        head_probs = F.softmax(head_logits, dim=-1)
        preds = head_probs.argmax(dim=-1).to(device)
        # Calculate custom loss if labels are provided
        # sentence stress detection
        loss = None
        loss_main = None
        if labels_head is not None:
            preds = torch.where(
                torch.isin(
                    labels_head, torch.tensor(list([-100])).to(device)  # 50257, 50362,
                ),
                torch.tensor(-100),
                preds,
            )
            # CrossEntropyLoss for the custom head
            loss_main = self.loss_fct(
                head_logits.reshape(-1, head_logits.size(-1)), labels_head.reshape(-1)
            )
            loss = self.lambda_ssd * loss_main

        # word stress loss
        loss_wsl = None
        if word_ids is not None and labels_head is not None and self.lambda_wsl > 0.0:
            loss_wsl = compute_adaptive_weighted_loss(head_logits, labels_head, word_ids)
            loss += self.lambda_wsl * loss_wsl

        return CustomPhnModelOutput(
            logits=head_logits,
            labels_head=labels_head,
            whisper_logits=backbone_outputs.logits,
            loss=loss,
            loss_main=loss_main,
            loss_wsl=loss_wsl,
            preds=preds,
        )

    def train(self, mode: Optional[bool] = True):
        super().train(mode)
        if self.freeze_pretrained_heads:
            self._freeze_pretrained_heads()
        for param in self.pos_bias.parameters():
            param.requires_grad = True
        self.pos_bias.train(mode)
        return self

    def __str__(self):
        return "WhiStressPos"

class WhiStressPhn(PreTrainedModel):
    
    config_class = WhisperConfig
    model_input_names = ["input_features", "labels_head", "whisper_labels"]

    def __init__(
        self,
        config: WhisperConfig,
        layer_for_head: Optional[int] = None,
        whisper_backbone_name="openai/whisper-small.en",
        class_weights = [1.0, 2.33],
        num_phones=39,
        loss_lambdas=None
    ):
        super().__init__(config=config, )
        self.whisper_backbone_name = whisper_backbone_name
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            self.whisper_backbone_name,
        ).eval()
        self.processor = WhisperProcessor.from_pretrained(self.whisper_backbone_name)

        input_dim = self.whisper_model.config.d_model  # Model's hidden size
        output_dim = 2  # Number of classes or output features for the new head

        config = self.whisper_model.config
        # add additional decoder block using the existing Whisper config
        self.additional_decoder_block = WhisperDecoderLayer(config)
        self.classifier = FCNN(input_dim, output_dim)
        # add additional decoder block using the torch embedding layer & transformer decoder
        self.phone_embed = nn.Embedding(num_embeddings=num_phones + 1, embedding_dim=config.d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.decoder_attention_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.phone_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.phone_stress_classifier = nn.Linear(config.d_model, 2)  # head for phone_stress
        
        # add weighted loss for CE
        class_weights = torch.tensor(class_weights)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
        self.phone_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.layer_for_head = -1 if layer_for_head is None else layer_for_head
        self.config = config

        if loss_lambdas:
            self.lambda_ssd = loss_lambdas["lambda_ssd"]
            self.lambda_wsd = loss_lambdas["lambda_wsd"]
            self.lambda_wsl = loss_lambdas["lambda_wsl"]
        else:
            self.lambda_ssd = 1
            self.lambda_wsd = -1
            self.lambda_wsl = -1

    def train(self, mode: Optional[bool] = True):
        super().train(mode)
        # freeze whisper and train task heads only
        self.whisper_model.eval()
        # mark whisper model requires grad false
        for param in self.whisper_model.parameters():
            param.requires_grad = False
        for param in self.additional_decoder_block.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        for param in self.phone_embed.parameters():
            param.requires_grad = True
        for param in self.phone_decoder.parameters():
            param.requires_grad = True
        for param in self.phone_stress_classifier.parameters():
            param.requires_grad = True
        
        self.additional_decoder_block.train(mode)
        self.classifier.train(mode)
        self.phone_embed.train(mode)
        self.phone_decoder.train(mode)
        self.phone_stress_classifier.train(mode)
        return self

    def forward(
        self,
        input_features,
        attention_mask=None,
        decoder_input_ids=None,
        labels_head=None,
        whisper_labels=None,
        phone_ids=None,
        phone_labels_head=None,
        token_pos_ids=None,
        word_ids=None
    ):  
        if phone_ids is None:
            raise ValueError("phone_ids is required for WhiStressPhn.forward")

        device = input_features.device
        self.whisper_model.eval()

        # pass the inputs through the model
        backbone_outputs = self.whisper_model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            labels=whisper_labels,
        )

        # Extract the hidden states of the last layer of the decoder
        decoder_last_layer_hidden_states = backbone_outputs.decoder_hidden_states[
            self.layer_for_head
        ].to(device)

        # Extract the hidden states of the layer of the encoder who encapsulates best the prosodic features
        layer_for_head_hidden_states = backbone_outputs.encoder_hidden_states[
            self.layer_for_head
        ].to(device)
        # Pass the decoder last hidden layers through the new head (decoder_block + lin cls)
        additional_decoder_block_outputs = self.additional_decoder_block(
            hidden_states=decoder_last_layer_hidden_states,
            encoder_hidden_states=layer_for_head_hidden_states,
        )[0].to(device)
        
        # pass the phone_ids through the embed layer
        phone_embed = self.phone_embed(phone_ids + 1)
        phone_decoder_block_outputs = self.phone_decoder(
                                        tgt=phone_embed,           # [B, T_phone, D]
                                        memory=layer_for_head_hidden_states  # [B, T_src, D]
                                        )

        # Sentence stress detection
        head_logits = self.classifier(additional_decoder_block_outputs)
        head_probs = F.softmax(head_logits, dim=-1)
        preds = head_probs.argmax(dim=-1).to(device)
        
        # Word stress detection
        phone_stress_logits = self.phone_stress_classifier(phone_decoder_block_outputs)
        phone_stress_probs = F.softmax(phone_stress_logits, dim=-1)
        phone_stress_preds = phone_stress_probs.argmax(dim=-1).to(device)

        # Calculate custom loss if labels are provided
        # sentence stress detection
        loss = None
        loss_main = None
        if labels_head is not None:
            preds = torch.where(
                torch.isin(
                    labels_head, torch.tensor(list([-100])).to(device)  # 50257, 50362,
                ),
                torch.tensor(-100),
                preds,
            )
            # CrossEntropyLoss for the custom head
            loss_main = self.loss_fct(
                head_logits.reshape(-1, head_logits.size(-1)), labels_head.reshape(-1)
            )
            loss = self.lambda_ssd * loss_main
        
        # word stress detection
        loss_wsd = None
        if phone_ids is not None and phone_labels_head is not None and self.lambda_wsd > 0.0:
            phone_stress_preds = torch.where(
                torch.isin(
                        phone_labels_head, torch.tensor(list([-100])).to(device)  # 50257, 50362,
                ),
                torch.tensor(-100),
                phone_stress_preds,
            )
            loss_wsd = self.phone_loss_fct(
                phone_stress_logits.reshape(-1, phone_stress_logits.size(-1)), phone_labels_head.reshape(-1)
            )
            loss += self.lambda_wsd * loss_wsd
        
        # word stress loss
        loss_wsl = None
        if word_ids is not None and labels_head is not None and self.lambda_wsl > 0.0:
            loss_wsl = compute_adaptive_weighted_loss(head_logits, labels_head, word_ids)
            loss += self.lambda_wsl * loss_wsl
        
        return CustomPhnModelOutput(
            logits=head_logits,
            labels_head=labels_head,
            phone_stress_logits=phone_stress_logits,
            whisper_logits=backbone_outputs.logits,
            loss=loss,
            loss_main=loss_main,
            loss_wsd=loss_wsd,
            loss_wsl=loss_wsl,
            preds=preds,
            phone_stress_preds=phone_stress_preds,
        )

    def generate(
        self,
        input_features,
        max_length=128,
        labels_head=None,
        whisper_labels=None,
        **generate_kwargs,
    ):
        """
        Generate both the Whisper output and custom head output sequences in alignment.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Generate the Whisper output sequence
        whisper_outputs = self.whisper_model.generate(
            input_features=input_features,
            max_length=max_length,
            labels=whisper_labels,
            do_sample=False,
            **generate_kwargs,
        )

        # pass the inputs through the model
        backbone_outputs = self.whisper_model(
            input_features=input_features,
            decoder_input_ids=whisper_outputs,
            output_hidden_states=True,
        )

        # Extract the hidden states of the last layer of the decoder
        decoder_last_layer_hidden_states = backbone_outputs.decoder_hidden_states[
            self.layer_for_head
        ].to(device)

        # Extract the hidden states of the last layer of the encoder
        layer_for_head_hidden_states = backbone_outputs.encoder_hidden_states[
            self.layer_for_head
        ].to(device)
        # Pass the decoder last hidden layers through the new head (decoder_block + lin cls)

        additional_decoder_block_outputs = self.additional_decoder_block(
            hidden_states=decoder_last_layer_hidden_states,
            encoder_hidden_states=layer_for_head_hidden_states,
        )
        head_logits = self.classifier(additional_decoder_block_outputs[0].to(device))
        # calculate softmax
        head_probs = F.softmax(head_logits, dim=-1)
        preds = head_probs.argmax(dim=-1).to(device)
        preds = torch.where(
            torch.isin(
                whisper_outputs, torch.tensor(list([50256])).to(device)  # 50257, 50362,
            ),
            torch.tensor(-100),
            preds,
        )
        return preds

    def generate_dual(
        self,
        input_features,
        attention_mask=None,
        max_length=200,
        labels_head=None,
        whisper_labels=None,
        **generate_kwargs,
    ):
        """
        Generate both the Whisper output and custom head output sequences in alignment.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Generate the Whisper output sequence
        whisper_outputs = self.whisper_model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            max_length=max_length,
            labels=whisper_labels,
            return_dict_in_generate=True,
            **generate_kwargs,
        )

        # pass the inputs through the model
        backbone_outputs = self.whisper_model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=whisper_outputs.sequences,
            output_hidden_states=True,
        )

        # Extract the hidden states of the last layer of the decoder
        decoder_last_layer_hidden_states = backbone_outputs.decoder_hidden_states[
            self.layer_for_head
        ].to(device)

        # Extract the hidden states of the last layer of the encoder
        layer_for_head_hidden_states = backbone_outputs.encoder_hidden_states[
            self.layer_for_head
        ].to(device)
        # Pass the decoder last hidden layers through the new head (decoder_block + lin cls)

        additional_decoder_block_outputs = self.additional_decoder_block(
            hidden_states=decoder_last_layer_hidden_states,
            encoder_hidden_states=layer_for_head_hidden_states,
        )
        head_logits = self.classifier(additional_decoder_block_outputs[0].to(device))
        head_probs = F.softmax(head_logits, dim=-1)
        preds = head_probs.argmax(dim=-1).to(device)
        preds = torch.where(
            torch.isin(
                whisper_outputs.sequences, torch.tensor(list([50256])).to(device)  # 50257, 50362,
            ),
            torch.tensor(-100),
            preds,
        )
        return CustomPhnModelOutput(
            logits=head_logits,
            head_preds=preds,
            whisper_logits=whisper_outputs.logits,
            preds=whisper_outputs.sequences
        )

    def __str__(self):
        return "WhiStressPhn"


class WhiStressPhnPairedResidual(WhiStressPhn):
    """Word-aligned bidirectional SSD/WSD residual coupling.

    The legacy WhiStressPhn path remains untouched. This class adds a
    zero-initialized paired residual between the SSD token representation
    and the WSD phone representation at the lexical-word level.
    """

    requires_word_alignment = True

    def __init__(
        self,
        config: WhisperConfig,
        layer_for_head: Optional[int] = None,
        whisper_backbone_name="openai/whisper-small.en",
        class_weights=[1.0, 2.33],
        num_phones=39,
        loss_lambdas=None,
        paired_residual_config=None,
        relation_loss_config=None,
        mil_loss_config=None,
    ):
        super().__init__(
            config=config,
            layer_for_head=layer_for_head,
            whisper_backbone_name=whisper_backbone_name,
            class_weights=class_weights,
            num_phones=num_phones,
            loss_lambdas=loss_lambdas,
        )

        paired_residual_config = paired_residual_config or {}
        relation_loss_config = relation_loss_config or {}
        mil_loss_config = mil_loss_config or {}

        d_model = self.config.d_model
        self.enable_wsd_to_ssd = bool(
            paired_residual_config.get("wsd_to_ssd", True)
        )
        self.enable_ssd_to_wsd = bool(
            paired_residual_config.get("ssd_to_wsd", True)
        )
        self.paired_temperature = float(
            paired_residual_config.get("temperature", 1.0)
        )
        if self.paired_temperature <= 0:
            raise ValueError("paired_residual temperature must be positive")

        residual_scale_init = float(
            paired_residual_config.get("residual_scale_init", 0.0)
        )

        # The two projections are deterministically overwritten with identity.
        # Restore the CPU RNG state afterwards so adding this architecture does
        # not change the subsequent DataLoader shuffle order for the same seed.
        cpu_rng_state = torch.get_rng_state()
        self.wsd_to_ssd_proj = nn.Linear(d_model, d_model, bias=False)
        self.ssd_to_wsd_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.wsd_to_ssd_proj.weight)
        nn.init.eye_(self.ssd_to_wsd_proj.weight)
        torch.set_rng_state(cpu_rng_state)

        # Zero residual scales make the initial coupled representations
        # identical to the uncoupled STRAW representations.

        self.wsd_to_ssd_scale = nn.Parameter(
            torch.tensor(residual_scale_init)
        )
        self.ssd_to_wsd_scale = nn.Parameter(
            torch.tensor(residual_scale_init)
        )

        if loss_lambdas:
            self.lambda_rank = float(loss_lambdas.get("lambda_rank", 0.0))
            self.lambda_mil = float(loss_lambdas.get("lambda_mil", 0.0))
        else:
            self.lambda_rank = 0.0
            self.lambda_mil = 0.0

        self.rank_margin = float(relation_loss_config.get("margin", 0.2))
        self.rank_temperature = float(
            relation_loss_config.get("temperature", 1.0)
        )
        if self.rank_temperature <= 0:
            raise ValueError("ranking temperature must be positive")
        self.mil_temperature = float(
            mil_loss_config.get("temperature", 1.0)
        )
        if self.mil_temperature <= 0:
            raise ValueError("MIL temperature must be positive")

    def train(self, mode: Optional[bool] = True):
        super().train(mode)

        for module in [self.wsd_to_ssd_proj, self.ssd_to_wsd_proj]:
            for param in module.parameters():
                param.requires_grad = True
            module.train(mode)

        self.wsd_to_ssd_scale.requires_grad = True
        self.ssd_to_wsd_scale.requires_grad = True
        return self

    def _build_paired_contexts(
        self,
        ssd_hidden_states,
        phone_hidden_states,
        preliminary_phone_logits,
        word_ids,
        phone_word_ids,
        phone_vowel_mask,
    ):
        """Build word-local contexts for both residual directions."""
        wsd_context_for_tokens = torch.zeros_like(ssd_hidden_states)
        ssd_context_for_phones = torch.zeros_like(phone_hidden_states)

        primary_scores = (
            preliminary_phone_logits[..., 1]
            - preliminary_phone_logits[..., 0]
        )

        for b in range(ssd_hidden_states.size(0)):
            valid_word_ids = torch.unique(word_ids[b][word_ids[b] >= 0])
            for wid in valid_word_ids:
                token_mask = word_ids[b] == wid
                phone_mask = phone_word_ids[b] == wid
                vowel_mask = phone_mask & phone_vowel_mask[b].bool()

                if not token_mask.any() or not phone_mask.any():
                    continue

                # SSD -> WSD: sentence-level word representation is injected
                # only into vowel phones, where lexical stress is realized.
                if self.enable_ssd_to_wsd and vowel_mask.any():
                    ssd_word_repr = ssd_hidden_states[b][token_mask].mean(dim=0)
                    ssd_context_for_phones[b][vowel_mask] = ssd_word_repr

                # WSD -> SSD: use a differentiable primary-stress-weighted
                # vowel anchor from the same lexical word.
                if self.enable_wsd_to_ssd and vowel_mask.any():
                    vowel_scores = primary_scores[b][vowel_mask]
                    weights = F.softmax(
                        vowel_scores / self.paired_temperature, dim=0
                    )
                    vowel_states = phone_hidden_states[b][vowel_mask]
                    wsd_word_repr = torch.sum(
                        weights.unsqueeze(-1) * vowel_states, dim=0
                    )
                    wsd_context_for_tokens[b][token_mask] = wsd_word_repr

        return wsd_context_for_tokens, ssd_context_for_phones

    def forward(
        self,
        input_features,
        attention_mask=None,
        decoder_input_ids=None,
        labels_head=None,
        whisper_labels=None,
        phone_ids=None,
        phone_labels_head=None,
        token_pos_ids=None,
        word_ids=None,
        phone_word_ids=None,
        phone_vowel_mask=None,
    ):
        if phone_ids is None:
            raise ValueError(
                "phone_ids is required for WhiStressPhnPairedResidual.forward"
            )
        if word_ids is None or phone_word_ids is None or phone_vowel_mask is None:
            raise ValueError(
                "word_ids, phone_word_ids, and phone_vowel_mask are required "
                "for paired residual coupling"
            )

        device = input_features.device
        self.whisper_model.eval()

        backbone_outputs = self.whisper_model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            labels=whisper_labels,
        )

        decoder_hidden_states = backbone_outputs.decoder_hidden_states[
            self.layer_for_head
        ].to(device)
        encoder_hidden_states = backbone_outputs.encoder_hidden_states[
            self.layer_for_head
        ].to(device)

        ssd_base = self.additional_decoder_block(
            hidden_states=decoder_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )[0].to(device)

        phone_embed = self.phone_embed(phone_ids + 1)
        phone_base = self.phone_decoder(
            tgt=phone_embed,
            memory=encoder_hidden_states,
        )

        # Preliminary WSD logits provide a differentiable estimate of the
        # primary lexical-stress locus used by the WSD -> SSD residual.
        preliminary_phone_logits = self.phone_stress_classifier(phone_base)

        (
            wsd_context_for_tokens,
            ssd_context_for_phones,
        ) = self._build_paired_contexts(
            ssd_hidden_states=ssd_base,
            phone_hidden_states=phone_base,
            preliminary_phone_logits=preliminary_phone_logits,
            word_ids=word_ids,
            phone_word_ids=phone_word_ids,
            phone_vowel_mask=phone_vowel_mask,
        )

        ssd_hidden_states = ssd_base
        if self.enable_wsd_to_ssd:
            ssd_hidden_states = (
                ssd_hidden_states
                + self.wsd_to_ssd_scale
                * self.wsd_to_ssd_proj(wsd_context_for_tokens)
            )

        phone_hidden_states = phone_base
        if self.enable_ssd_to_wsd:
            phone_hidden_states = (
                phone_hidden_states
                + self.ssd_to_wsd_scale
                * self.ssd_to_wsd_proj(ssd_context_for_phones)
            )

        head_logits = self.classifier(ssd_hidden_states)
        head_probs = F.softmax(head_logits, dim=-1)
        preds = head_probs.argmax(dim=-1).to(device)

        phone_stress_logits = self.phone_stress_classifier(phone_hidden_states)
        phone_stress_probs = F.softmax(phone_stress_logits, dim=-1)
        phone_stress_preds = phone_stress_probs.argmax(dim=-1).to(device)

        loss_terms = []
        loss_main = None
        loss_wsd = None
        loss_wsl = None
        loss_rank = None
        loss_mil = None

        if labels_head is not None:
            preds = torch.where(
                labels_head == -100,
                torch.tensor(-100, device=device),
                preds,
            )
            loss_main = self.loss_fct(
                head_logits.reshape(-1, head_logits.size(-1)),
                labels_head.reshape(-1),
            )
            if self.lambda_ssd > 0.0:
                loss_terms.append(self.lambda_ssd * loss_main)

        if (
            phone_labels_head is not None
            and self.lambda_wsd > 0.0
        ):
            phone_stress_preds = torch.where(
                phone_labels_head == -100,
                torch.tensor(-100, device=device),
                phone_stress_preds,
            )
            loss_wsd = self.phone_loss_fct(
                phone_stress_logits.reshape(-1, phone_stress_logits.size(-1)),
                phone_labels_head.reshape(-1),
            )
            loss_terms.append(self.lambda_wsd * loss_wsd)

        if (
            labels_head is not None
            and phone_labels_head is not None
            and self.lambda_rank > 0.0
        ):
            loss_rank = compute_cross_granularity_conditional_ranking_loss(
                ssd_hidden_states=ssd_base,
                phone_hidden_states=phone_base,
                phone_stress_logits=preliminary_phone_logits,
                labels_head=labels_head,
                word_ids=word_ids,
                phone_labels_head=phone_labels_head,
                phone_word_ids=phone_word_ids,
                phone_vowel_mask=phone_vowel_mask,
                margin=self.rank_margin,
                temperature=self.rank_temperature,
            )
            loss_terms.append(self.lambda_rank * loss_rank)

        if (
            labels_head is not None
            and self.lambda_mil > 0.0
        ):
            loss_mil = compute_word_level_mil_loss(
                logits=head_logits,
                labels_head=labels_head,
                word_ids=word_ids,
                temperature=self.mil_temperature,
            )
            loss_terms.append(self.lambda_mil * loss_mil)

        # WSL remains available for controlled legacy comparisons, but this
        # new model uses the corrected logit-aligned word_ids mapping.
        if (
            labels_head is not None
            and self.lambda_wsl > 0.0
        ):
            loss_wsl = compute_adaptive_weighted_loss(
                head_logits, labels_head, word_ids
            )
            loss_terms.append(self.lambda_wsl * loss_wsl)

        loss = torch.stack(loss_terms).sum() if loss_terms else None

        return CustomPhnModelOutput(
            logits=head_logits,
            labels_head=labels_head,
            phone_stress_logits=phone_stress_logits,
            whisper_logits=backbone_outputs.logits,
            loss=loss,
            loss_main=loss_main,
            loss_wsd=loss_wsd,
            loss_wsl=loss_wsl,
            loss_rank=loss_rank,
            loss_mil=loss_mil,
            preds=preds,
            phone_stress_preds=phone_stress_preds,
        )

    def __str__(self):
        return "WhiStressPhnPairedResidual"


class WhiStressPhnLocusCoupled(WhiStressPhn):
    """WSD-locus-guided SSD with word-level SSD-to-WSD residual coupling."""

    requires_word_alignment = True

    def __init__(
        self,
        config: WhisperConfig,
        layer_for_head: Optional[int] = None,
        whisper_backbone_name="openai/whisper-small.en",
        class_weights=[1.0, 2.33],
        num_phones=39,
        loss_lambdas=None,
        locus_coupling_config=None,
    ):
        super().__init__(
            config=config,
            layer_for_head=layer_for_head,
            whisper_backbone_name=whisper_backbone_name,
            class_weights=class_weights,
            num_phones=num_phones,
            loss_lambdas=loss_lambdas,
        )

        locus_coupling_config = locus_coupling_config or {}
        self.enable_wsd_to_ssd = bool(
            locus_coupling_config.get("wsd_to_ssd", True)
        )
        self.enable_ssd_to_wsd = bool(
            locus_coupling_config.get("ssd_to_wsd", True)
        )
        self.primary_temperature = float(
            locus_coupling_config.get("primary_temperature", 1.0)
        )
        self.locus_epsilon = float(
            locus_coupling_config.get("locus_epsilon", 1e-6)
        )
        if self.primary_temperature <= 0.0:
            raise ValueError("primary_temperature must be positive")
        if self.locus_epsilon <= 0.0:
            raise ValueError("locus_epsilon must be positive")

        self.wsd_to_ssd_scale = nn.Parameter(
            torch.tensor(
                float(
                    locus_coupling_config.get(
                        "wsd_to_ssd_scale_init", 0.0
                    )
                )
            )
        )
        self.ssd_to_wsd_scale = nn.Parameter(
            torch.tensor(
                float(
                    locus_coupling_config.get(
                        "ssd_to_wsd_scale_init", 0.0
                    )
                )
            )
        )

        # The projection is initialized to identity without changing the RNG
        # sequence used by the subsequent DataLoader shuffle.
        cpu_rng_state = torch.get_rng_state()
        self.ssd_to_wsd_proj = nn.Linear(
            self.config.d_model, self.config.d_model, bias=False
        )
        nn.init.eye_(self.ssd_to_wsd_proj.weight)
        torch.set_rng_state(cpu_rng_state)

    def train(self, mode: Optional[bool] = True):
        super().train(mode)
        for param in self.ssd_to_wsd_proj.parameters():
            param.requires_grad = True
        self.ssd_to_wsd_proj.train(mode)
        self.wsd_to_ssd_scale.requires_grad = True
        self.ssd_to_wsd_scale.requires_grad = True
        return self

    def _forward_phone_decoder_with_attention(
        self,
        phone_embed,
        encoder_hidden_states,
    ):
        """Run the existing phone decoder and expose its cross-attention."""
        hidden_states = phone_embed
        phone_cross_attention = None

        for layer in self.phone_decoder.layers:
            if layer.norm_first:
                hidden_states = hidden_states + layer._sa_block(
                    layer.norm1(hidden_states), None, None, False
                )
                cross_query = layer.norm2(hidden_states)
                cross_output, phone_cross_attention = layer.multihead_attn(
                    cross_query,
                    encoder_hidden_states,
                    encoder_hidden_states,
                    attn_mask=None,
                    key_padding_mask=None,
                    need_weights=True,
                    average_attn_weights=True,
                    is_causal=False,
                )
                hidden_states = hidden_states + layer.dropout2(cross_output)
                hidden_states = hidden_states + layer._ff_block(
                    layer.norm3(hidden_states)
                )
            else:
                hidden_states = layer.norm1(
                    hidden_states
                    + layer._sa_block(
                        hidden_states, None, None, False
                    )
                )
                cross_output, phone_cross_attention = layer.multihead_attn(
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states,
                    attn_mask=None,
                    key_padding_mask=None,
                    need_weights=True,
                    average_attn_weights=True,
                    is_causal=False,
                )
                hidden_states = layer.norm2(
                    hidden_states + layer.dropout2(cross_output)
                )
                hidden_states = layer.norm3(
                    hidden_states + layer._ff_block(hidden_states)
                )

        if self.phone_decoder.norm is not None:
            hidden_states = self.phone_decoder.norm(hidden_states)
        if phone_cross_attention is None:
            raise RuntimeError("phone_decoder must contain at least one layer")

        return hidden_states, phone_cross_attention

    def _build_ssd_cross_attention_bias(
        self,
        preliminary_phone_logits,
        phone_cross_attention,
        word_ids,
        phone_word_ids,
        phone_vowel_mask,
    ):
        """Map predicted primary-vowel attention to SSD token/audio bias."""
        batch_size, token_length = word_ids.shape
        audio_length = phone_cross_attention.size(-1)
        attention_bias = phone_cross_attention.new_zeros(
            batch_size, 1, token_length, audio_length
        )
        if not self.enable_wsd_to_ssd:
            return attention_bias

        primary_margins = (
            preliminary_phone_logits[..., 1]
            - preliminary_phone_logits[..., 0]
        )

        for b in range(batch_size):
            valid_word_ids = torch.unique(word_ids[b][word_ids[b] >= 0])
            for wid in valid_word_ids:
                token_mask = word_ids[b] == wid
                vowel_mask = (
                    (phone_word_ids[b] == wid)
                    & phone_vowel_mask[b].bool()
                )
                if not token_mask.any() or not vowel_mask.any():
                    continue

                primary_weights = F.softmax(
                    primary_margins[b][vowel_mask]
                    / self.primary_temperature,
                    dim=0,
                )
                word_locus_distribution = torch.sum(
                    primary_weights.unsqueeze(-1)
                    * phone_cross_attention[b][vowel_mask],
                    dim=0,
                )
                word_locus_distribution = word_locus_distribution.clamp_min(
                    self.locus_epsilon
                )
                attention_bias[b, 0, token_mask, :] = (
                    self.wsd_to_ssd_scale
                    * torch.log(word_locus_distribution)
                )

        return attention_bias

    def _build_ssd_to_wsd_residual(
        self,
        ssd_hidden_states,
        ssd_logits,
        phone_base,
        phone_cross_attention,
        encoder_hidden_states,
        word_ids,
        phone_word_ids,
        phone_vowel_mask,
    ):
        """Pool locus-guided SSD states as in PairedResidual."""
        del ssd_logits, phone_cross_attention, encoder_hidden_states
        ssd_context_for_phones = torch.zeros_like(phone_base)

        for b in range(ssd_hidden_states.size(0)):
            valid_word_ids = torch.unique(word_ids[b][word_ids[b] >= 0])
            for wid in valid_word_ids:
                token_mask = word_ids[b] == wid
                vowel_mask = (
                    (phone_word_ids[b] == wid)
                    & phone_vowel_mask[b].bool()
                )
                if not token_mask.any() or not vowel_mask.any():
                    continue

                ssd_word_repr = ssd_hidden_states[b][token_mask].mean(dim=0)
                ssd_context_for_phones[b][vowel_mask] = ssd_word_repr

        return ssd_context_for_phones

    def forward(
        self,
        input_features,
        attention_mask=None,
        decoder_input_ids=None,
        labels_head=None,
        whisper_labels=None,
        phone_ids=None,
        phone_labels_head=None,
        token_pos_ids=None,
        word_ids=None,
        phone_word_ids=None,
        phone_vowel_mask=None,
    ):
        del token_pos_ids
        if phone_ids is None:
            raise ValueError(f"phone_ids is required for {self.__class__.__name__}")
        if word_ids is None or phone_word_ids is None or phone_vowel_mask is None:
            raise ValueError(
                "word_ids, phone_word_ids, and phone_vowel_mask are required "
                "for locus coupling"
            )

        device = input_features.device
        self.whisper_model.eval()
        backbone_outputs = self.whisper_model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            labels=whisper_labels,
        )
        decoder_hidden_states = backbone_outputs.decoder_hidden_states[
            self.layer_for_head
        ].to(device)
        encoder_hidden_states = backbone_outputs.encoder_hidden_states[
            self.layer_for_head
        ].to(device)

        phone_embed = self.phone_embed(phone_ids + 1)
        phone_base, phone_cross_attention = (
            self._forward_phone_decoder_with_attention(
                phone_embed=phone_embed,
                encoder_hidden_states=encoder_hidden_states,
            )
        )
        preliminary_phone_logits = self.phone_stress_classifier(phone_base)

        ssd_cross_attention_bias = self._build_ssd_cross_attention_bias(
            preliminary_phone_logits=preliminary_phone_logits,
            phone_cross_attention=phone_cross_attention,
            word_ids=word_ids,
            phone_word_ids=phone_word_ids,
            phone_vowel_mask=phone_vowel_mask,
        )
        ssd_hidden_states = self.additional_decoder_block(
            hidden_states=decoder_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=ssd_cross_attention_bias,
        )[0].to(device)
        head_logits = self.classifier(ssd_hidden_states)

        phone_hidden_states = phone_base
        if self.enable_ssd_to_wsd:
            ssd_to_wsd_residual = self._build_ssd_to_wsd_residual(
                ssd_hidden_states=ssd_hidden_states,
                ssd_logits=head_logits,
                phone_base=phone_base,
                phone_cross_attention=phone_cross_attention,
                encoder_hidden_states=encoder_hidden_states,
                word_ids=word_ids,
                phone_word_ids=phone_word_ids,
                phone_vowel_mask=phone_vowel_mask,
            )
            phone_hidden_states = (
                phone_hidden_states
                + self.ssd_to_wsd_scale
                * self.ssd_to_wsd_proj(ssd_to_wsd_residual)
            )

        phone_stress_logits = self.phone_stress_classifier(phone_hidden_states)
        preds = F.softmax(head_logits, dim=-1).argmax(dim=-1).to(device)
        phone_stress_preds = (
            F.softmax(phone_stress_logits, dim=-1).argmax(dim=-1).to(device)
        )

        loss_terms = []
        loss_main = None
        loss_wsd = None
        loss_wsl = None

        if labels_head is not None:
            preds = torch.where(
                labels_head == -100,
                torch.tensor(-100, device=device),
                preds,
            )
            loss_main = self.loss_fct(
                head_logits.reshape(-1, head_logits.size(-1)),
                labels_head.reshape(-1),
            )
            if self.lambda_ssd > 0.0:
                loss_terms.append(self.lambda_ssd * loss_main)

        if phone_labels_head is not None and self.lambda_wsd > 0.0:
            phone_stress_preds = torch.where(
                phone_labels_head == -100,
                torch.tensor(-100, device=device),
                phone_stress_preds,
            )
            loss_wsd = self.phone_loss_fct(
                phone_stress_logits.reshape(-1, phone_stress_logits.size(-1)),
                phone_labels_head.reshape(-1),
            )
            loss_terms.append(self.lambda_wsd * loss_wsd)

        if labels_head is not None and self.lambda_wsl > 0.0:
            loss_wsl = compute_adaptive_weighted_loss(
                head_logits, labels_head, word_ids
            )
            loss_terms.append(self.lambda_wsl * loss_wsl)

        loss = torch.stack(loss_terms).sum() if loss_terms else None
        return CustomPhnModelOutput(
            logits=head_logits,
            labels_head=labels_head,
            phone_stress_logits=phone_stress_logits,
            whisper_logits=backbone_outputs.logits,
            loss=loss,
            loss_main=loss_main,
            loss_wsd=loss_wsd,
            loss_wsl=loss_wsl,
            loss_rank=None,
            loss_mil=None,
            preds=preds,
            phone_stress_preds=phone_stress_preds,
        )

    def __str__(self):
        return "WhiStressPhnLocusCoupled"


class WhiStressPhnLocusCoupledRealization(WhiStressPhnLocusCoupled):
    """Locus-guided SSD with SSD-modulated vowel acoustic realization."""

    def _build_ssd_to_wsd_residual(
        self,
        ssd_hidden_states,
        ssd_logits,
        phone_base,
        phone_cross_attention,
        encoder_hidden_states,
        word_ids,
        phone_word_ids,
        phone_vowel_mask,
    ):
        del ssd_hidden_states, phone_base
        phone_acoustic_context = torch.bmm(
            phone_cross_attention, encoder_hidden_states
        )
        word_prominence_for_phones = phone_acoustic_context.new_zeros(
            phone_acoustic_context.size(0),
            phone_acoustic_context.size(1),
            1,
        )
        token_stress_probs = F.softmax(ssd_logits, dim=-1)[..., 1]

        for b in range(ssd_logits.size(0)):
            valid_word_ids = torch.unique(word_ids[b][word_ids[b] >= 0])
            for wid in valid_word_ids:
                token_mask = word_ids[b] == wid
                vowel_mask = (
                    (phone_word_ids[b] == wid)
                    & phone_vowel_mask[b].bool()
                )
                if not token_mask.any() or not vowel_mask.any():
                    continue

                word_prominence = token_stress_probs[b][token_mask].mean()
                word_prominence_for_phones[b][vowel_mask] = word_prominence

        return phone_acoustic_context * word_prominence_for_phones

    def __str__(self):
        return "WhiStressPhnLocusCoupledRealization"


class WhiStressPhnIa(WhiStressPhn):
    
    config_class = WhisperConfig
    model_input_names = ["input_features", "labels_head", "whisper_labels"]

    def __init__(
        self,
        config: WhisperConfig,
        layer_for_head: Optional[int] = None,
        whisper_backbone_name="openai/whisper-small.en",
        class_weights = [1.0, 2.33],
        num_phones=39,
        loss_lambdas=None
    ):
        super().__init__(config=config, layer_for_head=layer_for_head, 
                        whisper_backbone_name=whisper_backbone_name, 
                        class_weights=class_weights, num_phones=num_phones, loss_lambdas=loss_lambdas)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.decoder_attention_heads,
            dim_feedforward=self.config.decoder_ffn_dim,
            dropout=self.config.dropout,
            batch_first=True
        )
        self.ssd_wsd_ia_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.mean_pool = MeanPooling()

    def train(self, mode: Optional[bool] = True):
        # freeze whisper and train classifier
        self.whisper_model.eval()
        # mark whisper model requires grad false
        for param in self.whisper_model.parameters():
            param.requires_grad = False
        for param in self.additional_decoder_block.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        for param in self.phone_embed.parameters():
            param.requires_grad = True
        for param in self.phone_decoder.parameters():
            param.requires_grad = True
        for param in self.phone_stress_classifier.parameters():
            param.requires_grad = True
        for param in self.ssd_wsd_ia_decoder.parameters():
            param.requires_grad = True
        
        self.additional_decoder_block.train()
        self.classifier.train()
        self.phone_embed.train()
        self.phone_decoder.train()
        self.phone_stress_classifier.train()
        self.ssd_wsd_ia_decoder.train()

    def forward(
        self,
        input_features,
        attention_mask=None,
        decoder_input_ids=None,
        labels_head=None,
        whisper_labels=None,
        phone_ids=None,
        phone_labels_head=None,
        token_pos_ids=None,
        word_ids=None
    ):  
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model.eval()

        # pass the inputs through the model
        backbone_outputs = self.whisper_model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            labels=whisper_labels,
        )

        # Extract the hidden states of the last layer of the decoder
        decoder_last_layer_hidden_states = backbone_outputs.decoder_hidden_states[
            self.layer_for_head
        ].to(device)

        # Extract the hidden states of the layer of the encoder who encapsulates best the prosodic features
        layer_for_head_hidden_states = backbone_outputs.encoder_hidden_states[
            self.layer_for_head
        ].to(device)
        # Pass the decoder last hidden layers through the new head (decoder_block + lin cls)
        additional_decoder_block_outputs = self.additional_decoder_block(
            hidden_states=decoder_last_layer_hidden_states,
            encoder_hidden_states=layer_for_head_hidden_states,
        )[0].to(device)
        
        # pass the phone_ids through the embed layer
        phone_embed = self.phone_embed(phone_ids + 1)
        phone_decoder_block_outputs = self.phone_decoder(
                                        tgt=phone_embed,           # [B, T_phone, D]
                                        memory=layer_for_head_hidden_states  # [B, T_src, D]
                                        )
        
        sentence_decoder_mask = ~torch.isin(decoder_input_ids, torch.tensor([50256, 50257, 50362], device=device))
        phone_decoder_mask = phone_ids != -1
        sentence_decoder_mask = sentence_decoder_mask.to(device)
        phone_decoder_mask = phone_decoder_mask.to(device)
        # [B, T_token, D] -> [B, D]
        sentence_decoder_vector, _ = self.mean_pool(additional_decoder_block_outputs, sentence_decoder_mask)
        # [B, T_phone, D] -> [B, D]
        phone_decoder_vector, _ = self.mean_pool(phone_decoder_block_outputs, phone_decoder_mask)
        # [B, 2, D]
        query_decoder_vector = torch.cat((sentence_decoder_vector.unsqueeze(1), phone_decoder_vector.unsqueeze(1)), dim=1)
        multi_granularity_vector = self.ssd_wsd_ia_decoder(
                                            tgt=query_decoder_vector,           # [B, 2, D]
                                            memory=layer_for_head_hidden_states  # [B, T_src, D]
                            )
        sentence_context_vector = multi_granularity_vector[:, 0:1, :]
        phone_context_vector = multi_granularity_vector[:, 1:, :]

        # Sentence stress detection
        head_logits = self.classifier(additional_decoder_block_outputs + sentence_context_vector)
        head_probs = F.softmax(head_logits, dim=-1)
        preds = head_probs.argmax(dim=-1).to(device)
        
        # Word stress detection
        phone_stress_logits = self.phone_stress_classifier(phone_decoder_block_outputs + phone_context_vector)
        phone_stress_probs = F.softmax(phone_stress_logits, dim=-1)
        phone_stress_preds = phone_stress_probs.argmax(dim=-1).to(device)

        # Calculate custom loss if labels are provided
        # sentence stress detection
        loss_main = None
        if labels_head is not None:
            preds = torch.where(
                torch.isin(
                    labels_head, torch.tensor(list([-100])).to(device)  # 50257, 50362,
                ),
                torch.tensor(-100),
                preds,
            )
            # CrossEntropyLoss for the custom head
            loss_main = self.loss_fct(
                head_logits.reshape(-1, head_logits.size(-1)), labels_head.reshape(-1)
            )
            loss = self.lambda_ssd * loss_main
        
        # word stress detection
        loss_phn = None 
        if phone_ids is not None and phone_labels_head is not None and self.lambda_wsd > 0.0:
            phone_stress_preds = torch.where(
                torch.isin(
                        phone_labels_head, torch.tensor(list([-100])).to(device)  # 50257, 50362,
                ),
                torch.tensor(-100),
                phone_stress_preds,
            )
            loss_wsd = self.phone_loss_fct(
                phone_stress_logits.reshape(-1, phone_stress_logits.size(-1)), phone_labels_head.reshape(-1)
            )
            loss += self.lambda_wsd * loss_wsd
        else:
            loss_wsd = None
        
        # word stress loss
        if word_ids is not None and labels_head is not None and self.lambda_wsl > 0.0:
            loss_wsl = compute_adaptive_weighted_loss(head_logits, labels_head, word_ids)
            loss += self.lambda_wsl * loss_wsl
        else:
            loss_wsl = None
        
        return CustomPhnModelOutput(
            logits=head_logits,
            labels_head=labels_head,
            phone_stress_logits=phone_stress_logits,
            whisper_logits=backbone_outputs.logits,
            loss=loss,
            loss_main=loss_main,
            loss_wsd=loss_wsd,
            loss_wsl=loss_wsl,
            preds=preds,
            phone_stress_preds=phone_stress_preds,
        )

    def __str__(self):
        return "WhiStressPhnIa"

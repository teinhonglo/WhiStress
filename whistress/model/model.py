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
from whistress.model.modules.net_utils import MeanPooling
from losses import compute_adaptive_weighted_loss


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
        self.whisper_model.eval()
        # mark whisper model requires grad false
        for param in self.whisper_model.parameters():
            param.requires_grad = False
        for param in self.additional_decoder_block.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        self.additional_decoder_block.train()
        self.classifier.train()

    def eval(self):
        self.whisper_model.eval()
        self.additional_decoder_block.eval()
        self.classifier.eval()

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

        if loss_lambdas:
            self.lambda_ssd = loss_lambdas["lambda_ssd"]
            self.lambda_wsd = loss_lambdas["lambda_wsd"]
            self.lambda_wsl = loss_lambdas["lambda_wsl"]
        else:
            self.lambda_ssd = 1
            self.lambda_wsd = -1
            self.lambda_wsl = -1

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
        
        self.additional_decoder_block.train()
        self.classifier.train()
        self.phone_embed.train()
        self.phone_decoder.train()
        self.phone_stress_classifier.train()

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
            d_model=config.d_model,
            nhead=config.decoder_attention_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
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

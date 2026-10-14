import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import T5EncoderModel, T5Config
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput


class T5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (..., d); d = config.d_model
        hidden_states = self.dropout(hidden_states)  # (..., d)
        hidden_states = self.dense(hidden_states)  # (..., d)
        hidden_states = torch.tanh(hidden_states)  # (..., d)
        hidden_states = self.dropout(hidden_states)  # (..., d)
        hidden_states = self.out_proj(hidden_states)  # (..., c), c = config.num_labels
        return hidden_states  # (..., c)


class T5ForSequenceClassification(T5EncoderModel):
    config_class = T5Config
    all_tied_weights_keys = {}
    def __init__(self, config: T5Config) -> None:
        super().__init__(config)
        self.classifier = T5ClassificationHead(config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        # input_ids: (b, l); attention_mask: (b, l); inputs_embeds: (b, l, d); head_mask: (h,) or (n_layers, h)
        # labels: (b,) for single-label classes, (b, c) for multi-label, (b,) or (b, c) for regression
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]  # (b, l, d)
        cls_token = sequence_output[:, 0, :]  # (b, d)

        logits = self.classifier(cls_token)  # (b, c)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


class T5ForTokenClassification(T5EncoderModel):
    config_class = T5Config
    all_tied_weights_keys = {}
    def __init__(self, config: T5Config) -> None:
        super().__init__(config)
        self.classifier = T5ClassificationHead(config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.Tensor | None = None,
    ) -> TokenClassifierOutput:
        # input_ids: (b, l); attention_mask: (b, l); inputs_embeds: (b, l, d); head_mask: (h,) or (n_layers, h)
        # labels: (b, l) for single-label classes, (b, l, c) for multi-label, (b, l) or (b, l, c) for regression
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]  # (b, l, d)

        logits = self.classifier(sequence_output)  # (b, l, c)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )

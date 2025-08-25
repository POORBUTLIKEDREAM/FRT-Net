# fault_diagnosis_project/models/components/Dual_Head.py
import torch.nn as nn


class DualHead(nn.Module):
    """
    Dual classifier head for main and auxiliary outputs.
    This version correctly accepts a single config object.
    """

    def __init__(self, config):
        """
        Initializes the DualHead.

        Args:
            config (EasyDict): The main configuration object for the project.
        """
        super().__init__()

        # Extract necessary parameters from the config object
        d_model = config.model.d_model
        num_classes = config.data.num_classes
        hidden_dims = config.model.classifier_hidden_dims
        dropout_rates = config.model.classifier_dropout_rates

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # --- Build the Main Classifier ---
        main_layers = []
        in_features = d_model
        # Dynamically create layers based on the config
        for i, hidden_dim in enumerate(hidden_dims):
            main_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rates[i])
            ])
            in_features = hidden_dim
        # Add the final output layer
        main_layers.append(nn.Linear(in_features, num_classes))
        self.main_classifier = nn.Sequential(nn.Flatten(), *main_layers)

        # --- Build the Auxiliary Classifier ---
        self.aux_classifier = nn.Sequential(
            nn.Linear(d_model, num_classes),
            nn.Dropout(0.1)
        )

    def forward(self, x_transformer_out):
        """
        Performs the forward pass for both classifier heads.

        Args:
            x_transformer_out (torch.Tensor): The output from the Transformer encoder.
        """
        # Main branch uses global average pooling on the transformer output
        x_pooled = self.global_avg_pool(x_transformer_out.permute(0, 2, 1))
        main_out = self.main_classifier(x_pooled)

        # Auxiliary branch uses the mean of all sequence tokens from the transformer output
        aux_out = self.aux_classifier(x_transformer_out.mean(dim=1))

        return main_out, aux_out

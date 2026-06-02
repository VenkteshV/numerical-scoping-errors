import toml
import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login
from peft import LoraConfig, get_peft_model


class CustomClassifier(torch.nn.Module):
    """
    Load the base LM and add classification head on top of it.
    """
    def __init__(self, model_name, num_labels=2, hidden_dim = None, dropout_value = 0.1, freeze_base_layer = True, use_lora = False, 
                 is_base_encoder = True, lora_rank = 8, lora_alpha = 16):
        
        
        super(CustomClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.is_base_encoder = is_base_encoder
        
        if freeze_base_layer:
            print("Freezing the base layers")
            for param in self.model.parameters():
                param.requires_grad = False
        
        if use_lora:
            print("Using LoRa for fine tuning")
            lora_config = LoraConfig(
                r = lora_rank, # The rank of the adapters
                lora_alpha = lora_alpha, # Scaling factor
                target_modules= ['q_proj', 'k_proj', 'v_proj'],
                lora_dropout=0,
                bias = 'none'
            )
            
            self.model = get_peft_model(self.model, lora_config)
        
        
        if hidden_dim is not None:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.model.config.hidden_size, hidden_dim),
                torch.nn.ReLU(), 
                torch.nn.Dropout(dropout_value),
                torch.nn.Linear(hidden_dim, num_labels)
            )
        elif hidden_dim is None:
            self.classifier = torch.nn.Linear(self.model.config.hidden_size, num_labels)
        
        

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        
        if self.is_base_encoder:
            pooled_output = outputs.pooler_output
        else: 
            pooled_output = outputs.last_hidden_state[:, -1, :]
        pooled_output = pooled_output.to(self.classifier.weight.dtype)
        logits = self.classifier(pooled_output)
        return logits


class VerifierHandler:
    """
    Generic verifier class for scoring pre-formatted input_text strings.

    Supports both fact-checking and numerical QA as long as the caller provides
    the correctly formatted input_text and model file.
    """

    def __init__(self, config_path: str):
        """
        Initialize Verifier with configuration and load all resources.
        """
        # Load config file (TOML format)
        self.config = toml.load(config_path)['configuration']

        # === Paths from config ===
        self.model_path = f"{self.config['output_dir']}"
        self.model_weights = f"{self.config['output_dir']}/model_weights"

        # === Device configuration (use GPU if available) ===
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # === HuggingFace authentication ===
        if "huggingface" in self.config and "access_token" in self.config["huggingface"]:
            login(token=self.config["huggingface"]["access_token"])

        # === Load tokenizer ===
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # === Initialize model with LoRA ===
        self.model = CustomClassifier(
            self.config["base_model"],
            num_labels=1,  # Binary classification
            hidden_dim=self.config.get("hidden_dim", None),
            dropout_value=self.config.get("dropout_rate", 0.0),
            is_base_encoder=self.config["is_base_encoder"],
            use_lora=self.config["use_lora"],
            lora_rank=self.config["lora_rank"],
            lora_alpha=self.config["lora_alpha"]
        )

        # === Load weights ===
        self.model.load_state_dict(
            torch.load(self.model_weights, map_location=self.device)
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Max length and batch size from config
        self.max_length = self.config["max_length"]
        self.batch_size = self.config["batch_size"]

    def score_text(self, input_text: str) -> float:
        """
        Score a single pre-formatted input text and return a verifier score in [0, 1]
        """

        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)

        score = torch.sigmoid(logits).item()
        return score
    
    def score_texts(self, input_texts: list[str]) -> list[float]:
        """
        Score a list of pre-formatted input texts with a batch size.
        """
        all_scores = list()

        for start_idx in range(0, len(input_texts), self.batch_size):
            batch_texts = input_texts[start_idx:start_idx + self.batch_size]
            encoding = self.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)

            scores = torch.sigmoid(logits).squeeze(-1).cpu().tolist()

            if isinstance(scores, float):
                scores = [scores]

            all_scores.extend(scores)
        return all_scores

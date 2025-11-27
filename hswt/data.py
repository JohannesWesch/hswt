"""Data module for HSWT training."""
import pytorch_lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Dict, Optional, List
import torch


class HSWTDataModule(L.LightningDataModule):
    """DataModule for HSWT training."""
    
    def __init__(
        self,
        dataset_name: str = "redpajama",
        tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
        max_length: int = 2048,
        batch_size: int = 32,
        num_workers: int = 4,
        phase: str = "supervised",  # "supervised" or "unsupervised"
        num_hierarchy_levels: int = 3,  # Number of hierarchy levels (including raw)
    ):
        """
        Args:
            dataset_name: Name of the dataset to load
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
            batch_size: Batch size
            num_workers: Number of data loader workers
            phase: Training phase ("supervised" or "unsupervised")
            num_hierarchy_levels: Number of hierarchy levels (raw + router levels)
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.phase = phase
        self.num_hierarchy_levels = num_hierarchy_levels
        
        self.tokenizer = None
        self.train_dataset = None
    
    def setup(self, stage: str):
        """Load and preprocess dataset."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        if stage == "fit" or stage is None:
            self.train_dataset = load_dataset(self.dataset_name, split="train", streaming=True)
    
    def _preprocess_batch(self, batch: Dict) -> Dict:
        """
        For Phase 1: Pre-calculate sentence/section boundaries.
        
        Args:
            batch: Raw batch from dataset
        
        Returns:
            Processed batch with boundary_masks list
        """
        # Tokenize text
        texts = batch.get("text", [])
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # For Phase 1 (supervised), pre-calculate boundary masks
        if self.phase == "supervised":
            boundary_masks = []
            # Generate boundary masks for each hierarchy level (except raw)
            # For now, use simple heuristics (periods, newlines)
            # In practice, you'd use SpaCy/NLTK for sentence detection
            for level in range(1, self.num_hierarchy_levels):
                # Simple heuristic: mark sentence boundaries (periods)
                # This is a placeholder - replace with proper NLP tools
                boundary_mask = self._detect_boundaries(input_ids, level)
                boundary_masks.append(boundary_mask)
            
            result["boundary_masks"] = boundary_masks
        
        return result
    
    def _detect_boundaries(self, input_ids: torch.Tensor, level: int) -> torch.Tensor:
        """
        Detect boundaries for a given hierarchy level.
        
        This is a placeholder implementation. In practice, use:
        - Level 1: Sentence boundaries (periods, exclamation, question marks)
        - Level 2: Section boundaries (paragraph breaks, headings)
        - etc.
        
        Args:
            input_ids: (B, L) token IDs
            level: Hierarchy level (1 = sentence, 2 = section, etc.)
        
        Returns:
            boundary_mask: (B, L) binary mask
        """
        B, L = input_ids.shape
        boundary_mask = torch.zeros(B, L, dtype=torch.bool)
        
        # Placeholder: mark every 20th token as boundary for level 1
        # Replace with proper NLP-based detection
        if level == 1:
            # Simple heuristic: look for period tokens (tokenizer-dependent)
            # This is just a placeholder
            period_token_id = self.tokenizer.encode(".", add_special_tokens=False)[0]
            boundary_mask = (input_ids == period_token_id)
        
        return boundary_mask
    
    def train_dataloader(self):
        """Return training DataLoader."""
        def collate_fn(batch):
            # Convert streaming dataset batch to dict
            if isinstance(batch, list):
                # Handle streaming dataset format
                texts = [item.get("text", "") for item in batch]
                batch_dict = {"text": texts}
            else:
                batch_dict = batch
            
            return self._preprocess_batch(batch_dict)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )


"""
Model architecture for text-to-LLVM IR translation.
Uses T5-small for optimal balance of accuracy and speed.
"""

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config
)
import torch


class TextToLLVMModel:
    """
    Wrapper for T5 model optimized for text-to-LLVM IR translation.
    T5-small provides good translation accuracy with fast training/inference.
    """
    
    def __init__(self, model_name: str = "t5-small", max_length: int = 512):
        """
        Initialize the model.
        
        Args:
            model_name: Base model to use (t5-small is optimal for this task)
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
    
    def preprocess(self, text: str, llvm_ir: str = None):
        """
        Preprocess input text and optionally target LLVM IR.
        
        Args:
            text: Input text description
            llvm_ir: Target LLVM IR code (for training)
            
        Returns:
            Dictionary with tokenized inputs
        """
        # Add task prefix for T5
        input_text = f"translate to llvm: {text}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        result = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        
        # Tokenize target if provided (for training)
        if llvm_ir is not None:
            targets = self.tokenizer(
                llvm_ir,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            result["labels"] = targets["input_ids"]
        
        return result
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            labels: Target labels (for training)
            
        Returns:
            Model outputs
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        if labels is not None:
            labels = labels.to(self.device)
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, text: str, num_beams: int = 4, max_new_tokens: int = 512):
        """
        Generate LLVM IR from input text.
        
        Args:
            text: Input text description
            num_beams: Number of beams for beam search
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated LLVM IR code
        """
        self.model.eval()
        
        # Preprocess input
        inputs = self.preprocess(text)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def save(self, output_dir: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    
    def load(self, model_dir: str):
        """Load model and tokenizer from directory."""
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.model.to(self.device)
        print(f"Model loaded from {model_dir}")
    
    def get_model(self):
        """Get the underlying model."""
        return self.model
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer


def create_model(model_name: str = "t5-small", max_length: int = 512):
    """
    Factory function to create a TextToLLVMModel.
    
    Args:
        model_name: Base model name
        max_length: Maximum sequence length
        
    Returns:
        TextToLLVMModel instance
    """
    return TextToLLVMModel(model_name=model_name, max_length=max_length)

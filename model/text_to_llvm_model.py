"""
Model architecture for text-to-LLVM IR translation.
Uses T5-small for optimal balance of accuracy and speed.
Optimized for best quality AI in least time with:
- Mixed precision training (FP16)
- Gradient checkpointing for memory efficiency
- Optimized generation parameters
- Model compilation support
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
    
    def __init__(self, model_name: str = "t5-small", max_length: int = 512, 
                 use_gradient_checkpointing: bool = True, compile_model: bool = False):
        """
        Initialize the model with optimization features.
        
        Args:
            model_name: Base model to use (t5-small is optimal for this task)
            max_length: Maximum sequence length
            use_gradient_checkpointing: Enable gradient checkpointing to save memory
            compile_model: Use torch.compile for faster inference (PyTorch 2.0+)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Initialize tokenizer with optimized settings
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            model_max_length=max_length
        )
        
        # Initialize model
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Enable gradient checkpointing for memory efficiency during training
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.model.to(self.device)
        
        # Compile model for faster inference (PyTorch 2.0+)
        if compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("Model compiled with torch.compile for faster inference")
            except Exception as e:
                print(f"Warning: Could not compile model: {e}")
        
        # Cache for repeated inputs
        self._generation_cache = {}
    
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
    
    def generate(self, text: str, num_beams: int = 5, max_new_tokens: int = 512,
                 temperature: float = 0.7, top_k: int = 50, top_p: float = 0.95,
                 use_cache: bool = True, repetition_penalty: float = 1.2):
        """
        Generate LLVM IR from input text with optimized parameters.
        
        Args:
            text: Input text description
            num_beams: Number of beams for beam search (higher = better quality, slower)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            use_cache: Use caching for repeated inputs
            repetition_penalty: Penalty for repeating tokens
            
        Returns:
            Generated LLVM IR code
        """
        # Check cache for repeated inputs
        cache_key = (text, num_beams, max_new_tokens, temperature)
        if use_cache and cache_key in self._generation_cache:
            return self._generation_cache[cache_key]
        
        self.model.eval()
        
        # Preprocess input
        inputs = self.preprocess(text)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Generate with optimized parameters
        with torch.no_grad():
            # Use mixed precision for faster inference
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=False if num_beams > 1 else True,  # Use sampling for greedy, beam for quality
                    early_stopping=True,
                    no_repeat_ngram_size=3,  # Prevent repetition
                    repetition_penalty=repetition_penalty,
                    length_penalty=1.0,  # Neutral length preference
                    num_return_sequences=1
                )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Cache result
        if use_cache and len(self._generation_cache) < 100:  # Limit cache size
            self._generation_cache[cache_key] = generated_text
        
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
        
        # Re-enable gradient checkpointing if needed
        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.model.to(self.device)
        print(f"Model loaded from {model_dir}")
    
    def get_model(self):
        """Get the underlying model."""
        return self.model
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer


def create_model(model_name: str = "t5-small", max_length: int = 512,
                 use_gradient_checkpointing: bool = True, compile_model: bool = False):
    """
    Factory function to create an optimized TextToLLVMModel.
    
    Args:
        model_name: Base model name
        max_length: Maximum sequence length
        use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        compile_model: Use torch.compile for faster inference
        
    Returns:
        Optimized TextToLLVMModel instance
    """
    return TextToLLVMModel(
        model_name=model_name,
        max_length=max_length,
        use_gradient_checkpointing=use_gradient_checkpointing,
        compile_model=compile_model
    )

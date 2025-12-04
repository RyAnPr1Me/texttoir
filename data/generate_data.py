"""
Quick data generation script for text-to-LLVM IR training pairs.
Generates small dataset for testing. Use generate_data_large.py for full 1GB dataset.
"""

import json
import random
import os
from typing import List, Tuple


class LLVMIRGenerator:
    """Generates diverse text descriptions and corresponding LLVM IR code."""
    
    def __init__(self):
        self.function_names = ['calculate', 'process', 'compute', 'transform', 'execute']
        self.var_names = ['x', 'y', 'z', 'a', 'b', 'result', 'temp', 'value']
        
    def generate_simple_arithmetic(self) -> List[Tuple[str, str]]:
        """Generate simple arithmetic operations."""
        examples = []
        
        # Addition
        examples.append((
            "Write a function that adds two integers and returns the result",
            """define i32 @add(i32 %a, i32 %b) {
entry:
  %result = add i32 %a, %b
  ret i32 %result
}"""
        ))
        
        # Subtraction
        examples.append((
            "Create a function to subtract one integer from another",
            """define i32 @subtract(i32 %a, i32 %b) {
entry:
  %result = sub i32 %a, %b
  ret i32 %result
}"""
        ))
        
        # Multiplication
        examples.append((
            "Implement a function that multiplies two integers",
            """define i32 @multiply(i32 %a, i32 %b) {
entry:
  %result = mul i32 %a, %b
  ret i32 %result
}"""
        ))
        
        # Division
        examples.append((
            "Write a function to divide two integers",
            """define i32 @divide(i32 %a, i32 %b) {
entry:
  %result = sdiv i32 %a, %b
  ret i32 %result
}"""
        ))
        
        return examples
    
    def generate_conditionals(self) -> List[Tuple[str, str]]:
        """Generate conditional logic examples."""
        examples = []
        
        # Simple if-else
        examples.append((
            "Create a function that returns the maximum of two integers",
            """define i32 @max(i32 %a, i32 %b) {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  ret i32 %a

if.else:
  ret i32 %b
}"""
        ))
        
        # Absolute value
        examples.append((
            "Implement a function that returns the absolute value of an integer",
            """define i32 @abs(i32 %x) {
entry:
  %cmp = icmp slt i32 %x, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %neg = sub i32 0, %x
  ret i32 %neg

if.else:
  ret i32 %x
}"""
        ))
        
        # Sign function
        examples.append((
            "Write a function that returns 1 if positive, -1 if negative, 0 if zero",
            """define i32 @sign(i32 %x) {
entry:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %positive, label %check.negative

positive:
  ret i32 1

check.negative:
  %cmp2 = icmp slt i32 %x, 0
  br i1 %cmp2, label %negative, label %zero

negative:
  ret i32 -1

zero:
  ret i32 0
}"""
        ))
        
        return examples
    
    def generate_loops(self) -> List[Tuple[str, str]]:
        """Generate loop examples."""
        examples = []
        
        # Sum from 1 to n
        examples.append((
            "Create a function that calculates the sum of integers from 1 to n",
            """define i32 @sum_to_n(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 1, %entry ], [ %i.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %sum.next = add i32 %sum, %i
  %i.next = add i32 %i, 1
  %cmp = icmp sle i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %sum.next
}"""
        ))
        
        # Factorial
        examples.append((
            "Implement a function to calculate factorial of n",
            """define i32 @factorial(i32 %n) {
entry:
  %cmp = icmp sle i32 %n, 1
  br i1 %cmp, label %base, label %loop.preheader

base:
  ret i32 1

loop.preheader:
  br label %loop

loop:
  %i = phi i32 [ %n, %loop.preheader ], [ %i.next, %loop ]
  %result = phi i32 [ 1, %loop.preheader ], [ %result.next, %loop ]
  %result.next = mul i32 %result, %i
  %i.next = sub i32 %i, 1
  %cmp.loop = icmp sgt i32 %i.next, 1
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret i32 %result.next
}"""
        ))
        
        return examples
    
    def generate_arrays(self) -> List[Tuple[str, str]]:
        """Generate array operation examples."""
        examples = []
        
        # Array sum
        examples.append((
            "Write a function that sums all elements in an integer array",
            """define i32 @array_sum(i32* %arr, i32 %len) {
entry:
  %cmp = icmp eq i32 %len, 0
  br i1 %cmp, label %exit, label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %ptr = getelementptr i32, i32* %arr, i32 %i
  %val = load i32, i32* %ptr
  %sum.next = add i32 %sum, %val
  %i.next = add i32 %i, 1
  %cmp.loop = icmp slt i32 %i.next, %len
  br i1 %cmp.loop, label %loop, label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  ret i32 %result
}"""
        ))
        
        return examples
    
    def generate_functions(self) -> List[Tuple[str, str]]:
        """Generate function call examples."""
        examples = []
        
        # Function composition
        examples.append((
            "Create a function that calls a square helper function",
            """define i32 @square(i32 %x) {
entry:
  %result = mul i32 %x, %x
  ret i32 %result
}

define i32 @sum_of_squares(i32 %a, i32 %b) {
entry:
  %sq_a = call i32 @square(i32 %a)
  %sq_b = call i32 @square(i32 %b)
  %result = add i32 %sq_a, %sq_b
  ret i32 %result
}"""
        ))
        
        return examples
    
    def generate_floating_point(self) -> List[Tuple[str, str]]:
        """Generate floating point examples."""
        examples = []
        
        examples.append((
            "Write a function that adds two floating point numbers",
            """define float @fadd(float %a, float %b) {
entry:
  %result = fadd float %a, %b
  ret float %result
}"""
        ))
        
        examples.append((
            "Create a function to multiply two floating point numbers",
            """define double @fmul(double %a, double %b) {
entry:
  %result = fmul double %a, %b
  ret double %result
}"""
        ))
        
        return examples
    
    def generate_logical_ops(self) -> List[Tuple[str, str]]:
        """Generate logical operation examples."""
        examples = []
        
        examples.append((
            "Implement a function that performs bitwise AND on two integers",
            """define i32 @bitwise_and(i32 %a, i32 %b) {
entry:
  %result = and i32 %a, %b
  ret i32 %result
}"""
        ))
        
        examples.append((
            "Write a function for bitwise OR operation",
            """define i32 @bitwise_or(i32 %a, i32 %b) {
entry:
  %result = or i32 %a, %b
  ret i32 %result
}"""
        ))
        
        examples.append((
            "Create a function that performs XOR on two integers",
            """define i32 @bitwise_xor(i32 %a, i32 %b) {
entry:
  %result = xor i32 %a, %b
  ret i32 %result
}"""
        ))
        
        return examples
    
    def generate_comparison(self) -> List[Tuple[str, str]]:
        """Generate comparison examples."""
        examples = []
        
        examples.append((
            "Write a function that checks if two integers are equal",
            """define i1 @is_equal(i32 %a, i32 %b) {
entry:
  %result = icmp eq i32 %a, %b
  ret i1 %result
}"""
        ))
        
        examples.append((
            "Create a function to check if first integer is greater than second",
            """define i1 @is_greater(i32 %a, i32 %b) {
entry:
  %result = icmp sgt i32 %a, %b
  ret i1 %result
}"""
        ))
        
        return examples
    
    def generate_all(self) -> List[Tuple[str, str]]:
        """Generate all training examples."""
        all_examples = []
        all_examples.extend(self.generate_simple_arithmetic())
        all_examples.extend(self.generate_conditionals())
        all_examples.extend(self.generate_loops())
        all_examples.extend(self.generate_arrays())
        all_examples.extend(self.generate_functions())
        all_examples.extend(self.generate_floating_point())
        all_examples.extend(self.generate_logical_ops())
        all_examples.extend(self.generate_comparison())
        return all_examples


def generate_augmented_examples(base_examples: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Augment examples with variations in wording."""
    augmented = []
    
    variations = {
        "Write": ["Implement", "Create", "Define", "Generate"],
        "Create": ["Write", "Implement", "Build", "Make"],
        "function": ["function", "subroutine", "procedure", "method"],
        "that": ["which", "that"],
        "returns": ["gives", "returns", "outputs"],
    }
    
    for text, ir in base_examples:
        augmented.append((text, ir))
        
        # Create variations
        for _ in range(2):
            new_text = text
            for old, new_options in variations.items():
                if old in new_text:
                    new_text = new_text.replace(old, random.choice(new_options), 1)
            if new_text != text:
                augmented.append((new_text, ir))
    
    return augmented


def save_dataset(examples: List[Tuple[str, str]], output_dir: str, split: str):
    """Save dataset in JSONL format."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{split}.jsonl")
    
    with open(output_file, 'w') as f:
        for text, ir in examples:
            example = {
                "text": text,
                "llvm_ir": ir
            }
            f.write(json.dumps(example) + '\n')
    
    print(f"Saved {len(examples)} examples to {output_file}")


def main():
    """Generate and save training data (small test dataset)."""
    print("=" * 60)
    print("Quick Text-to-LLVM IR Dataset Generation")
    print("NOTE: For 1GB production dataset, use generate_data_large.py")
    print("=" * 60)
    
    # Generate base examples
    generator = LLVMIRGenerator()
    base_examples = generator.generate_all()
    print(f"Generated {len(base_examples)} base examples")
    
    # Augment with variations
    all_examples = generate_augmented_examples(base_examples)
    print(f"Total examples after augmentation: {len(all_examples)}")
    
    # Shuffle
    random.shuffle(all_examples)
    
    # Split into train/val/test
    total = len(all_examples)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    
    train_data = all_examples[:train_size]
    val_data = all_examples[train_size:train_size + val_size]
    test_data = all_examples[train_size + val_size:]
    
    # Save datasets
    output_dir = "dataset"
    save_dataset(train_data, output_dir, "train")
    save_dataset(val_data, output_dir, "val")
    save_dataset(test_data, output_dir, "test")
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print(f"Train: {len(train_data)} examples")
    print(f"Val: {len(val_data)} examples")
    print(f"Test: {len(test_data)} examples")
    print("=" * 60)


if __name__ == "__main__":
    print("\n💡 TIP: For full 1GB dataset with extreme diversity,")
    print("         use: python data/generate_data_large.py\n")
    main()

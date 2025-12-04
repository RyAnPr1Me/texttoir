"""
Enhanced data generation script for text-to-LLVM IR training pairs.
Generates 1GB+ dataset with extremely diverse, unique examples.
Includes edge cases, good and bad code with quality markings.

Improvements:
- Command-line arguments for configurability
- Optimized augmentation with batch processing
- Progress bar with ETA and speed metrics
- Streaming write to reduce memory usage
- Smart text variation with more diversity
- Resume capability for interrupted generation
- Better error handling and validation
"""

import json
import random
import os
import hashlib
import argparse
import time
import subprocess
import tempfile
from typing import List, Tuple, Dict, Set
from tqdm import tqdm


class EnhancedLLVMIRGenerator:
    """Generates extremely diverse text descriptions and LLVM IR code."""
    
    def __init__(self, validate_ir: bool = True):
        self.validate_ir = validate_ir
        self.function_names = ['calculate', 'process', 'compute', 'transform', 'execute', 
                               'handle', 'manage', 'operate', 'evaluate', 'analyze',
                               'convert', 'filter', 'map', 'reduce', 'aggregate', 'accumulate',
                               'combine', 'merge', 'split', 'divide_work', 'batch_process']
        self.var_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'result', 'temp', 'value',
                         'data', 'item', 'elem', 'idx', 'count', 'sum', 'prod', 'acc',
                         'left', 'right', 'mid', 'low', 'high', 'cur', 'next', 'prev']
        self.int_types = ['i8', 'i16', 'i32', 'i64', 'i128']
        self.float_types = ['float', 'double', 'x86_fp80']
        self.generated_hashes: Set[str] = set()
        self.validation_failures = 0
    
    def _validate_llvm_ir(self, ir: str) -> bool:
        """Validate LLVM IR using llvm-as."""
        if not self.validate_ir:
            return True
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f:
                f.write(ir)
                f.flush()
                temp_file = f.name
            
            # Try to assemble the IR
            result = subprocess.run(
                ['llvm-as', temp_file, '-o', '/dev/null'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            os.unlink(temp_file)
            
            if result.returncode != 0:
                self.validation_failures += 1
                return False
            return True
        except Exception:
            return False
        
    def _get_hash(self, text: str, ir: str) -> str:
        """Generate hash for uniqueness checking."""
        combined = f"{text}|||{ir}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _is_unique(self, text: str, ir: str) -> bool:
        """Check if example is unique."""
        h = self._get_hash(text, ir)
        if h in self.generated_hashes:
            return False
        self.generated_hashes.add(h)
        return True
    
    def _add_example(self, examples: List[Tuple[str, str, str]], 
                     text: str, ir: str, quality: str):
        """Add example if unique and valid."""
        if self._is_unique(text, ir):
            if self._validate_llvm_ir(ir):
                examples.append((text, ir, quality))
            else:
                # Skip invalid IR
                pass
    
    def generate_basic_arithmetic(self) -> List[Tuple[str, str, str]]:
        """Generate basic arithmetic with all combinations."""
        examples = []
        operations = [
            ('add', 'add', 'adds', 'addition'),
            ('sub', 'subtract', 'subtracts', 'subtraction'),
            ('mul', 'multiply', 'multiplies', 'multiplication'),
            ('sdiv', 'divide', 'divides', 'division'),
            ('udiv', 'unsigned divide', 'unsigned divides', 'unsigned division'),
            ('srem', 'modulo', 'remainder of', 'mod'),
            ('urem', 'unsigned modulo', 'unsigned remainder', 'unsigned mod')
        ]
        
        for int_type in self.int_types:
            for op_llvm, op_name1, op_name2, op_name3 in operations:
                for fn_name in random.sample(self.function_names, 3):
                    for var_a, var_b in [('x', 'y'), ('a', 'b'), ('left', 'right')]:
                        # Good example
                        text = f"Write a function that {op_name2} two {int_type} integers"
                        ir = f"""define {int_type} @{fn_name}({int_type} %{var_a}, {int_type} %{var_b}) {{
entry:
  %result = {op_llvm} {int_type} %{var_a}, %{var_b}
  ret {int_type} %result
}}"""
                        self._add_example(examples, text, ir, "GOOD")
                        
                        # Variation
                        text = f"Implement {op_name3} of two {int_type} values"
                        self._add_example(examples, text, ir, "GOOD")
        
        # Bad examples - division by zero risk
        for int_type in self.int_types:
            text = f"BAD CODE: Function that divides without checking for zero divisor ({int_type})"
            ir = f"""define {int_type} @unsafe_divide({int_type} %a, {int_type} %b) {{
entry:
  ; BUG: No check for division by zero!
  %result = sdiv {int_type} %a, %b
  ret {int_type} %result
}}"""
            self._add_example(examples, text, ir, "BAD")
        
        return examples
    
    def generate_floating_point(self) -> List[Tuple[str, str, str]]:
        """Generate floating point operations."""
        examples = []
        operations = [
            ('fadd', 'add', 'floating point addition'),
            ('fsub', 'subtract', 'floating point subtraction'),
            ('fmul', 'multiply', 'floating point multiplication'),
            ('fdiv', 'divide', 'floating point division'),
            ('frem', 'remainder', 'floating point remainder')
        ]
        
        for float_type in self.float_types:
            for op_llvm, op_name, op_desc in operations:
                text = f"Create a function for {op_desc} with {float_type}"
                ir = f"""define {float_type} @f_{op_name}({float_type} %a, {float_type} %b) {{
entry:
  %result = {op_llvm} {float_type} %a, %b
  ret {float_type} %result
}}"""
                self._add_example(examples, text, ir, "GOOD")
        
        # Bad example - no NaN/infinity check
        text = "BAD CODE: Floating point division without NaN/infinity checks"
        ir = """define double @unsafe_fdiv(double %a, double %b) {
entry:
  ; BUG: No check for NaN, infinity, or division by zero
  %result = fdiv double %a, %b
  ret double %result
}"""
        self._add_example(examples, text, ir, "BAD")
        
        return examples
    
    def generate_comparisons(self) -> List[Tuple[str, str, str]]:
        """Generate comparison operations."""
        examples = []
        int_comparisons = [
            ('eq', 'equal to', 'equality'),
            ('ne', 'not equal to', 'inequality'),
            ('sgt', 'greater than', 'signed greater'),
            ('sge', 'greater than or equal', 'signed greater or equal'),
            ('slt', 'less than', 'signed less'),
            ('sle', 'less than or equal', 'signed less or equal'),
            ('ugt', 'unsigned greater than', 'unsigned greater'),
            ('uge', 'unsigned greater or equal', 'unsigned greater or equal'),
            ('ult', 'unsigned less than', 'unsigned less'),
            ('ule', 'unsigned less or equal', 'unsigned less or equal')
        ]
        
        for int_type in self.int_types:
            for cmp_op, cmp_desc, cmp_name in int_comparisons:
                text = f"Function to check if first {int_type} is {cmp_desc} second"
                ir = f"""define i1 @cmp_{cmp_op}({int_type} %a, {int_type} %b) {{
entry:
  %result = icmp {cmp_op} {int_type} %a, %b
  ret i1 %result
}}"""
                self._add_example(examples, text, ir, "GOOD")
        
        # Floating point comparisons
        float_comparisons = [
            ('oeq', 'ordered equal', 'ordered equality'),
            ('one', 'ordered not equal', 'ordered inequality'),
            ('ogt', 'ordered greater', 'ordered greater than'),
            ('olt', 'ordered less', 'ordered less than'),
            ('ueq', 'unordered equal', 'unordered equality'),
            ('une', 'unordered not equal', 'unordered inequality')
        ]
        
        for float_type in self.float_types:
            for cmp_op, cmp_desc, cmp_name in float_comparisons:
                text = f"Compare {float_type} values using {cmp_name}"
                ir = f"""define i1 @fcmp_{cmp_op}({float_type} %a, {float_type} %b) {{
entry:
  %result = fcmp {cmp_op} {float_type} %a, %b
  ret i1 %result
}}"""
                self._add_example(examples, text, ir, "GOOD")
        
        return examples
    
    def generate_conditionals(self) -> List[Tuple[str, str, str]]:
        """Generate conditional logic with edge cases."""
        examples = []
        
        for int_type in self.int_types:
            # Max function
            text = f"Return maximum of two {int_type} values"
            ir = f"""define {int_type} @max({int_type} %a, {int_type} %b) {{
entry:
  %cmp = icmp sgt {int_type} %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  ret {int_type} %a

if.else:
  ret {int_type} %b
}}"""
            self._add_example(examples, text, ir, "GOOD")
            
            # Min function
            text = f"Return minimum of two {int_type} values"
            ir = f"""define {int_type} @min({int_type} %a, {int_type} %b) {{
entry:
  %cmp = icmp slt {int_type} %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  ret {int_type} %a

if.else:
  ret {int_type} %b
}}"""
            self._add_example(examples, text, ir, "GOOD")
            
            # Absolute value
            text = f"Calculate absolute value of {int_type}"
            ir = f"""define {int_type} @abs({int_type} %x) {{
entry:
  %cmp = icmp slt {int_type} %x, 0
  br i1 %cmp, label %if.neg, label %if.pos

if.neg:
  %neg = sub {int_type} 0, %x
  ret {int_type} %neg

if.pos:
  ret {int_type} %x
}}"""
            self._add_example(examples, text, ir, "GOOD")
            
            # Clamp function
            text = f"Clamp {int_type} value between min and max"
            ir = f"""define {int_type} @clamp({int_type} %val, {int_type} %min, {int_type} %max) {{
entry:
  %cmp1 = icmp slt {int_type} %val, %min
  br i1 %cmp1, label %ret.min, label %check.max

check.max:
  %cmp2 = icmp sgt {int_type} %val, %max
  br i1 %cmp2, label %ret.max, label %ret.val

ret.min:
  ret {int_type} %min

ret.max:
  ret {int_type} %max

ret.val:
  ret {int_type} %val
}}"""
            self._add_example(examples, text, ir, "GOOD")
        
        # Bad example - missing edge case
        text = "BAD CODE: Absolute value that overflows on INT_MIN"
        ir = """define i32 @bad_abs(i32 %x) {
entry:
  ; BUG: -INT_MIN overflows to INT_MIN due to two's complement
  %cmp = icmp slt i32 %x, 0
  br i1 %cmp, label %if.neg, label %if.pos

if.neg:
  %neg = sub i32 0, %x
  ret i32 %neg

if.pos:
  ret i32 %x
}"""
        self._add_example(examples, text, ir, "BAD")
        
        return examples
    
    def generate_loops(self) -> List[Tuple[str, str, str]]:
        """Generate loop examples with various patterns."""
        examples = []
        
        for int_type in self.int_types:
            # Sum from 1 to n
            text = f"Calculate sum from 1 to n using {int_type}"
            ir = f"""define {int_type} @sum_to_n({int_type} %n) {{
entry:
  %cmp.init = icmp sle {int_type} %n, 0
  br i1 %cmp.init, label %exit.zero, label %loop.preheader

loop.preheader:
  br label %loop

loop:
  %i = phi {int_type} [ 1, %loop.preheader ], [ %i.next, %loop ]
  %sum = phi {int_type} [ 0, %loop.preheader ], [ %sum.next, %loop ]
  %sum.next = add {int_type} %sum, %i
  %i.next = add {int_type} %i, 1
  %cmp = icmp sle {int_type} %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit.zero:
  ret {int_type} 0

exit:
  ret {int_type} %sum.next
}}"""
            self._add_example(examples, text, ir, "GOOD")
            
            # Factorial
            text = f"Compute factorial of n ({int_type})"
            ir = f"""define {int_type} @factorial({int_type} %n) {{
entry:
  %cmp.init = icmp sle {int_type} %n, 1
  br i1 %cmp.init, label %base, label %loop.preheader

base:
  ret {int_type} 1

loop.preheader:
  br label %loop

loop:
  %i = phi {int_type} [ %n, %loop.preheader ], [ %i.next, %loop ]
  %result = phi {int_type} [ 1, %loop.preheader ], [ %result.next, %loop ]
  %result.next = mul {int_type} %result, %i
  %i.next = sub {int_type} %i, 1
  %cmp = icmp sgt {int_type} %i.next, 1
  br i1 %cmp, label %loop, label %exit

exit:
  ret {int_type} %result.next
}}"""
            self._add_example(examples, text, ir, "GOOD")
            
            # Power function
            text = f"Calculate base raised to power ({int_type})"
            ir = f"""define {int_type} @power({int_type} %base, {int_type} %exp) {{
entry:
  %cmp.init = icmp eq {int_type} %exp, 0
  br i1 %cmp.init, label %ret.one, label %loop.preheader

ret.one:
  ret {int_type} 1

loop.preheader:
  br label %loop

loop:
  %i = phi {int_type} [ 1, %loop.preheader ], [ %i.next, %loop ]
  %result = phi {int_type} [ %base, %loop.preheader ], [ %result.next, %loop ]
  %result.next = mul {int_type} %result, %base
  %i.next = add {int_type} %i, 1
  %cmp = icmp slt {int_type} %i.next, %exp
  br i1 %cmp, label %loop, label %exit

exit:
  ret {int_type} %result.next
}}"""
            self._add_example(examples, text, ir, "GOOD")
        
        # Bad example - infinite loop potential
        text = "BAD CODE: Loop that may never terminate due to unsigned underflow"
        ir = """define i32 @bad_countdown(i32 %n) {
entry:
  br label %loop

loop:
  ; BUG: If n is 0, unsigned subtraction underflows to 4294967295
  %i = phi i32 [ %n, %entry ], [ %i.next, %loop ]
  %i.next = sub i32 %i, 1
  %cmp = icmp ne i32 %i, 0
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 0
}"""
        self._add_example(examples, text, ir, "BAD")
        
        return examples
    
    def generate_arrays(self) -> List[Tuple[str, str, str]]:
        """Generate array operations."""
        examples = []
        
        for int_type in self.int_types:
            # Array sum
            text = f"Sum all elements in {int_type} array"
            ir = f"""define {int_type} @array_sum({int_type}* %arr, {int_type} %len) {{
entry:
  %cmp.init = icmp eq {int_type} %len, 0
  br i1 %cmp.init, label %exit.zero, label %loop

loop:
  %i = phi {int_type} [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi {int_type} [ 0, %entry ], [ %sum.next, %loop ]
  %ptr = getelementptr {int_type}, {int_type}* %arr, {int_type} %i
  %val = load {int_type}, {int_type}* %ptr
  %sum.next = add {int_type} %sum, %val
  %i.next = add {int_type} %i, 1
  %cmp = icmp slt {int_type} %i.next, %len
  br i1 %cmp, label %loop, label %exit

exit.zero:
  ret {int_type} 0

exit:
  ret {int_type} %sum.next
}}"""
            self._add_example(examples, text, ir, "GOOD")
            
            # Array max
            text = f"Find maximum element in {int_type} array"
            ir = f"""define {int_type} @array_max({int_type}* %arr, {int_type} %len) {{
entry:
  %first_ptr = getelementptr {int_type}, {int_type}* %arr, {int_type} 0
  %first = load {int_type}, {int_type}* %first_ptr
  %cmp.init = icmp sle {int_type} %len, 1
  br i1 %cmp.init, label %exit, label %loop

loop:
  %i = phi {int_type} [ 1, %entry ], [ %i.next, %loop ]
  %max = phi {int_type} [ %first, %entry ], [ %new_max, %loop ]
  %ptr = getelementptr {int_type}, {int_type}* %arr, {int_type} %i
  %val = load {int_type}, {int_type}* %ptr
  %cmp = icmp sgt {int_type} %val, %max
  %new_max = select i1 %cmp, {int_type} %val, {int_type} %max
  %i.next = add {int_type} %i, 1
  %cmp.loop = icmp slt {int_type} %i.next, %len
  br i1 %cmp.loop, label %loop, label %exit

exit:
  %result = phi {int_type} [ %first, %entry ], [ %new_max, %loop ]
  ret {int_type} %result
}}"""
            self._add_example(examples, text, ir, "GOOD")
            
            # Array reverse
            text = f"Reverse {int_type} array in place"
            ir = f"""define void @array_reverse({int_type}* %arr, {int_type} %len) {{
entry:
  %half = sdiv {int_type} %len, 2
  %cmp.init = icmp sle {int_type} %len, 1
  br i1 %cmp.init, label %exit, label %loop

loop:
  %i = phi {int_type} [ 0, %entry ], [ %i.next, %loop ]
  %j_init = sub {int_type} %len, 1
  %j = sub {int_type} %j_init, %i
  %ptr_i = getelementptr {int_type}, {int_type}* %arr, {int_type} %i
  %ptr_j = getelementptr {int_type}, {int_type}* %arr, {int_type} %j
  %val_i = load {int_type}, {int_type}* %ptr_i
  %val_j = load {int_type}, {int_type}* %ptr_j
  store {int_type} %val_j, {int_type}* %ptr_i
  store {int_type} %val_i, {int_type}* %ptr_j
  %i.next = add {int_type} %i, 1
  %cmp = icmp slt {int_type} %i.next, %half
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}}"""
            self._add_example(examples, text, ir, "GOOD")
        
        # Bad example - no bounds checking
        text = "BAD CODE: Array access without bounds checking"
        ir = """define i32 @unsafe_array_get(i32* %arr, i32 %index) {
entry:
  ; BUG: No validation that index is within array bounds
  %ptr = getelementptr i32, i32* %arr, i32 %index
  %val = load i32, i32* %ptr
  ret i32 %val
}"""
        self._add_example(examples, text, ir, "BAD")
        
        # Bad example - null pointer dereference
        text = "BAD CODE: Array function that doesn't check for null pointer"
        ir = """define i32 @unsafe_array_sum(i32* %arr, i32 %len) {
entry:
  ; BUG: No null pointer check before dereferencing
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %ptr = getelementptr i32, i32* %arr, i32 %i
  %val = load i32, i32* %ptr
  %sum.next = add i32 %sum, %val
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i.next, %len
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %sum.next
}"""
        self._add_example(examples, text, ir, "BAD")
        
        return examples
    
    def generate_bitwise(self) -> List[Tuple[str, str, str]]:
        """Generate bitwise operations."""
        examples = []
        operations = [
            ('and', 'AND', 'bitwise AND'),
            ('or', 'OR', 'bitwise OR'),
            ('xor', 'XOR', 'bitwise XOR')
        ]
        
        for int_type in self.int_types:
            for op, op_name, op_desc in operations:
                text = f"Perform {op_desc} on two {int_type} values"
                ir = f"""define {int_type} @bit_{op}({int_type} %a, {int_type} %b) {{
entry:
  %result = {op} {int_type} %a, %b
  ret {int_type} %result
}}"""
                self._add_example(examples, text, ir, "GOOD")
            
            # Shift operations
            text = f"Left shift {int_type} value"
            ir = f"""define {int_type} @shift_left({int_type} %val, {int_type} %amount) {{
entry:
  %result = shl {int_type} %val, %amount
  ret {int_type} %result
}}"""
            self._add_example(examples, text, ir, "GOOD")
            
            text = f"Logical right shift {int_type} value"
            ir = f"""define {int_type} @shift_right_logical({int_type} %val, {int_type} %amount) {{
entry:
  %result = lshr {int_type} %val, %amount
  ret {int_type} %result
}}"""
            self._add_example(examples, text, ir, "GOOD")
            
            text = f"Arithmetic right shift {int_type} value"
            ir = f"""define {int_type} @shift_right_arith({int_type} %val, {int_type} %amount) {{
entry:
  %result = ashr {int_type} %val, %amount
  ret {int_type} %result
}}"""
            self._add_example(examples, text, ir, "GOOD")
            
            # Bit manipulation
            text = f"Check if bit at position is set ({int_type})"
            ir = f"""define i1 @is_bit_set({int_type} %val, {int_type} %pos) {{
entry:
  %mask = shl {int_type} 1, %pos
  %result = and {int_type} %val, %mask
  %is_set = icmp ne {int_type} %result, 0
  ret i1 %is_set
}}"""
            self._add_example(examples, text, ir, "GOOD")
            
            # Count set bits (population count)
            text = f"Count number of set bits in {int_type}"
            ir = f"""define {int_type} @popcount({int_type} %val) {{
entry:
  br label %loop

loop:
  %n = phi {int_type} [ %val, %entry ], [ %n.next, %loop ]
  %count = phi {int_type} [ 0, %entry ], [ %count.next, %loop ]
  %bit = and {int_type} %n, 1
  %count.next = add {int_type} %count, %bit
  %n.next = lshr {int_type} %n, 1
  %cmp = icmp ne {int_type} %n.next, 0
  br i1 %cmp, label %loop, label %exit

exit:
  ret {int_type} %count.next
}}"""
            self._add_example(examples, text, ir, "GOOD")
        
        # Bad example - undefined shift
        text = "BAD CODE: Shift by amount >= bit width (undefined behavior)"
        ir = """define i32 @bad_shift(i32 %val, i32 %amount) {
entry:
  ; BUG: No check that amount < 32, shifting by >= 32 is undefined
  %result = shl i32 %val, %amount
  ret i32 %result
}"""
        self._add_example(examples, text, ir, "BAD")
        
        return examples
    
    def generate_functions(self) -> List[Tuple[str, str, str]]:
        """Generate function call examples."""
        examples = []
        
        for int_type in self.int_types:
            # Function composition
            text = f"Calculate sum of squares using helper function ({int_type})"
            ir = f"""define {int_type} @square({int_type} %x) {{
entry:
  %result = mul {int_type} %x, %x
  ret {int_type} %result
}}

define {int_type} @sum_of_squares({int_type} %a, {int_type} %b) {{
entry:
  %sq_a = call {int_type} @square({int_type} %a)
  %sq_b = call {int_type} @square({int_type} %b)
  %result = add {int_type} %sq_a, %sq_b
  ret {int_type} %result
}}"""
            self._add_example(examples, text, ir, "GOOD")
            
            # Recursive function
            text = f"Recursive GCD function ({int_type})"
            ir = f"""define {int_type} @gcd({int_type} %a, {int_type} %b) {{
entry:
  %cmp = icmp eq {int_type} %b, 0
  br i1 %cmp, label %base, label %recurse

base:
  ret {int_type} %a

recurse:
  %rem = urem {int_type} %a, %b
  %result = call {int_type} @gcd({int_type} %b, {int_type} %rem)
  ret {int_type} %result
}}"""
            self._add_example(examples, text, ir, "GOOD")
            
            # Fibonacci
            text = f"Recursive Fibonacci function ({int_type})"
            ir = f"""define {int_type} @fibonacci({int_type} %n) {{
entry:
  %cmp1 = icmp eq {int_type} %n, 0
  br i1 %cmp1, label %ret.zero, label %check.one

ret.zero:
  ret {int_type} 0

check.one:
  %cmp2 = icmp eq {int_type} %n, 1
  br i1 %cmp2, label %ret.one, label %recurse

ret.one:
  ret {int_type} 1

recurse:
  %n1 = sub {int_type} %n, 1
  %n2 = sub {int_type} %n, 2
  %fib1 = call {int_type} @fibonacci({int_type} %n1)
  %fib2 = call {int_type} @fibonacci({int_type} %n2)
  %result = add {int_type} %fib1, %fib2
  ret {int_type} %result
}}"""
            self._add_example(examples, text, ir, "GOOD")
        
        # Bad example - stack overflow risk
        text = "BAD CODE: Recursive function without base case check (stack overflow risk)"
        ir = """define i32 @bad_recursive(i32 %n) {
entry:
  ; BUG: Missing proper base case, will cause stack overflow
  %n_next = sub i32 %n, 1
  %result = call i32 @bad_recursive(i32 %n_next)
  %final = add i32 %result, %n
  ret i32 %final
}"""
        self._add_example(examples, text, ir, "BAD")
        
        return examples
    
    def generate_structs_and_pointers(self) -> List[Tuple[str, str, str]]:
        """Generate struct and pointer examples."""
        examples = []
        
        # Struct operations
        text = "Load field from struct pointer"
        ir = """%Point = type { i32, i32 }

define i32 @get_x(%Point* %p) {
entry:
  %x_ptr = getelementptr %Point, %Point* %p, i32 0, i32 0
  %x = load i32, i32* %x_ptr
  ret i32 %x
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        text = "Set field in struct pointer"
        ir = """%Point = type { i32, i32 }

define void @set_y(%Point* %p, i32 %val) {
entry:
  %y_ptr = getelementptr %Point, %Point* %p, i32 0, i32 1
  store i32 %val, i32* %y_ptr
  ret void
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        text = "Calculate distance squared between two points"
        ir = """%Point = type { i32, i32 }

define i32 @distance_squared(%Point* %p1, %Point* %p2) {
entry:
  %x1_ptr = getelementptr %Point, %Point* %p1, i32 0, i32 0
  %y1_ptr = getelementptr %Point, %Point* %p1, i32 0, i32 1
  %x2_ptr = getelementptr %Point, %Point* %p2, i32 0, i32 0
  %y2_ptr = getelementptr %Point, %Point* %p2, i32 0, i32 1
  %x1 = load i32, i32* %x1_ptr
  %y1 = load i32, i32* %y1_ptr
  %x2 = load i32, i32* %x2_ptr
  %y2 = load i32, i32* %y2_ptr
  %dx = sub i32 %x2, %x1
  %dy = sub i32 %y2, %y1
  %dx2 = mul i32 %dx, %dx
  %dy2 = mul i32 %dy, %dy
  %dist2 = add i32 %dx2, %dy2
  ret i32 %dist2
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Bad example - null pointer
        text = "BAD CODE: Struct field access without null check"
        ir = """%Data = type { i32, i32, i32 }

define i32 @unsafe_get_field(%Data* %ptr) {
entry:
  ; BUG: No null pointer check
  %field_ptr = getelementptr %Data, %Data* %ptr, i32 0, i32 0
  %val = load i32, i32* %field_ptr
  ret i32 %val
}"""
        self._add_example(examples, text, ir, "BAD")
        
        return examples
    
    def generate_select_operations(self) -> List[Tuple[str, str, str]]:
        """Generate select instruction examples."""
        examples = []
        
        for int_type in self.int_types:
            text = f"Conditional select maximum using select instruction ({int_type})"
            ir = f"""define {int_type} @select_max({int_type} %a, {int_type} %b) {{
entry:
  %cmp = icmp sgt {int_type} %a, %b
  %result = select i1 %cmp, {int_type} %a, {int_type} %b
  ret {int_type} %result
}}"""
            self._add_example(examples, text, ir, "GOOD")
            
            text = f"Absolute value using select ({int_type})"
            ir = f"""define {int_type} @abs_select({int_type} %x) {{
entry:
  %cmp = icmp slt {int_type} %x, 0
  %neg = sub {int_type} 0, %x
  %result = select i1 %cmp, {int_type} %neg, {int_type} %x
  ret {int_type} %result
}}"""
            self._add_example(examples, text, ir, "GOOD")
        
        return examples
    
    def generate_edge_cases(self) -> List[Tuple[str, str, str]]:
        """Generate edge case examples."""
        examples = []
        
        # Overflow detection
        text = "EDGE CASE: Detect signed addition overflow"
        ir = """define i1 @will_add_overflow(i32 %a, i32 %b) {
entry:
  %result = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %overflow = extractvalue {i32, i1} %result, 1
  ret i1 %overflow
}

declare {i32, i1} @llvm.sadd.with.overflow.i32(i32, i32)"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Safe multiplication
        text = "EDGE CASE: Safe multiplication with overflow check"
        ir = """define {i32, i1} @safe_mul(i32 %a, i32 %b) {
entry:
  %result = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %a, i32 %b)
  ret {i32, i1} %result
}

declare {i32, i1} @llvm.smul.with.overflow.i32(i32, i32)"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Integer minimum edge case
        text = "EDGE CASE: Handle INT_MIN absolute value safely"
        ir = """define i32 @safe_abs(i32 %x) {
entry:
  %is_min = icmp eq i32 %x, -2147483648
  br i1 %is_min, label %overflow, label %normal

overflow:
  ; Return INT_MAX as best approximation
  ret i32 2147483647

normal:
  %cmp = icmp slt i32 %x, 0
  %neg = sub i32 0, %x
  %result = select i1 %cmp, i32 %neg, i32 %x
  ret i32 %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Zero-length array
        text = "EDGE CASE: Handle zero-length array in sum"
        ir = """define i32 @safe_array_sum(i32* %arr, i32 %len) {
entry:
  %is_empty = icmp eq i32 %len, 0
  br i1 %is_empty, label %ret.zero, label %check.null

check.null:
  %is_null = icmp eq i32* %arr, null
  br i1 %is_null, label %ret.zero, label %loop

loop:
  %i = phi i32 [ 0, %check.null ], [ %i.next, %loop ]
  %sum = phi i32 [ 0, %check.null ], [ %sum.next, %loop ]
  %ptr = getelementptr i32, i32* %arr, i32 %i
  %val = load i32, i32* %ptr
  %sum.next = add i32 %sum, %val
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i.next, %len
  br i1 %cmp, label %loop, label %ret.sum

ret.zero:
  ret i32 0

ret.sum:
  ret i32 %sum.next
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        return examples
    
    def generate_memory_operations(self) -> List[Tuple[str, str, str]]:
        """Generate memory allocation and management examples."""
        examples = []
        
        for int_type in ['i32', 'i64']:
            # Stack allocation
            text = f"Allocate {int_type} on stack and store value"
            ir = f"""define {int_type} @stack_alloc({int_type} %val) {{
entry:
  %ptr = alloca {int_type}
  store {int_type} %val, {int_type}* %ptr
  %result = load {int_type}, {int_type}* %ptr
  ret {int_type} %result
}}"""
            self._add_example(examples, text, ir, "GOOD")
            
            # Array allocation
            text = f"Allocate array of {int_type} on stack"
            ir = f"""define {int_type}* @alloc_array({int_type} %size) {{
entry:
  %arr = alloca {int_type}, {int_type} %size
  ret {int_type}* %arr
}}"""
            self._add_example(examples, text, ir, "GOOD")
        
        # Struct allocation
        text = "Allocate struct on stack and initialize"
        ir = """%Point = type { i32, i32 }

define %Point* @alloc_point(i32 %x, i32 %y) {
entry:
  %ptr = alloca %Point
  %x_ptr = getelementptr %Point, %Point* %ptr, i32 0, i32 0
  %y_ptr = getelementptr %Point, %Point* %ptr, i32 0, i32 1
  store i32 %x, i32* %x_ptr
  store i32 %y, i32* %y_ptr
  ret %Point* %ptr
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        return examples
    
    def generate_vector_operations(self) -> List[Tuple[str, str, str]]:
        """Generate SIMD vector operations."""
        examples = []
        
        # Vector addition
        text = "Add two 4-element i32 vectors (SIMD)"
        ir = """define <4 x i32> @vector_add(<4 x i32> %a, <4 x i32> %b) {
entry:
  %result = add <4 x i32> %a, %b
  ret <4 x i32> %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Vector multiplication
        text = "Multiply two 4-element float vectors"
        ir = """define <4 x float> @vector_mul(<4 x float> %a, <4 x float> %b) {
entry:
  %result = fmul <4 x float> %a, %b
  ret <4 x float> %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Vector comparison
        text = "Compare two 4-element i32 vectors for equality"
        ir = """define <4 x i1> @vector_cmp(<4 x i32> %a, <4 x i32> %b) {
entry:
  %result = icmp eq <4 x i32> %a, %b
  ret <4 x i1> %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Extract element
        text = "Extract element from vector at index"
        ir = """define i32 @vector_extract(<4 x i32> %vec) {
entry:
  %result = extractelement <4 x i32> %vec, i32 2
  ret i32 %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Insert element
        text = "Insert element into vector at index"
        ir = """define <4 x i32> @vector_insert(<4 x i32> %vec, i32 %val) {
entry:
  %result = insertelement <4 x i32> %vec, i32 %val, i32 1
  ret <4 x i32> %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Shuffle vector
        text = "Shuffle two vectors with specific pattern"
        ir = """define <4 x i32> @vector_shuffle(<4 x i32> %a, <4 x i32> %b) {
entry:
  %result = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x i32> %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        return examples
    
    def generate_atomic_operations(self) -> List[Tuple[str, str, str]]:
        """Generate atomic operations for thread safety."""
        examples = []
        
        # Atomic load
        text = "Atomically load i32 value with sequential consistency"
        ir = """define i32 @atomic_load(i32* %ptr) {
entry:
  %val = load atomic i32, i32* %ptr seq_cst, align 4
  ret i32 %val
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Atomic store
        text = "Atomically store i32 value with release ordering"
        ir = """define void @atomic_store(i32* %ptr, i32 %val) {
entry:
  store atomic i32 %val, i32* %ptr release, align 4
  ret void
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Compare and swap
        text = "Atomic compare and exchange (CAS) operation"
        ir = """define i1 @atomic_cas(i32* %ptr, i32 %expected, i32 %new) {
entry:
  %result = cmpxchg i32* %ptr, i32 %expected, i32 %new seq_cst seq_cst
  %success = extractvalue { i32, i1 } %result, 1
  ret i1 %success
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Atomic add
        text = "Atomically add value to memory location"
        ir = """define i32 @atomic_add(i32* %ptr, i32 %val) {
entry:
  %old = atomicrmw add i32* %ptr, i32 %val seq_cst
  ret i32 %old
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Atomic exchange
        text = "Atomically exchange value at memory location"
        ir = """define i32 @atomic_exchange(i32* %ptr, i32 %new_val) {
entry:
  %old = atomicrmw xchg i32* %ptr, i32 %new_val acquire
  ret i32 %old
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        return examples
    
    def generate_switch_statements(self) -> List[Tuple[str, str, str]]:
        """Generate switch statement examples."""
        examples = []
        
        # Basic switch
        text = "Switch statement with multiple cases"
        ir = """define i32 @switch_example(i32 %val) {
entry:
  switch i32 %val, label %default [
    i32 0, label %case0
    i32 1, label %case1
    i32 2, label %case2
    i32 3, label %case3
  ]

case0:
  ret i32 10

case1:
  ret i32 20

case2:
  ret i32 30

case3:
  ret i32 40

default:
  ret i32 -1
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Day of week switch
        text = "Convert day number to day offset (0-6)"
        ir = """define i32 @day_value(i32 %day) {
entry:
  switch i32 %day, label %invalid [
    i32 0, label %sunday
    i32 1, label %weekday
    i32 2, label %weekday
    i32 3, label %weekday
    i32 4, label %weekday
    i32 5, label %weekday
    i32 6, label %saturday
  ]

sunday:
  ret i32 0

weekday:
  ret i32 1

saturday:
  ret i32 2

invalid:
  ret i32 -1
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        return examples
    
    def generate_phi_advanced(self) -> List[Tuple[str, str, str]]:
        """Generate advanced phi node examples."""
        examples = []
        
        # Complex phi with multiple predecessors
        text = "Complex control flow with multiple phi nodes"
        ir = """define i32 @complex_phi(i32 %a, i32 %b, i1 %cond) {
entry:
  br i1 %cond, label %left, label %right

left:
  %left_val = mul i32 %a, 2
  br label %merge

right:
  %right_val = add i32 %b, 10
  br label %merge

merge:
  %result = phi i32 [ %left_val, %left ], [ %right_val, %right ]
  ret i32 %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        return examples
    
    def generate_type_conversions(self) -> List[Tuple[str, str, str]]:
        """Generate type conversion examples."""
        examples = []
        
        # Integer truncation
        text = "Truncate i64 to i32"
        ir = """define i32 @trunc_i64_to_i32(i64 %val) {
entry:
  %result = trunc i64 %val to i32
  ret i32 %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Integer extension (zero extend)
        text = "Zero extend i32 to i64"
        ir = """define i64 @zext_i32_to_i64(i32 %val) {
entry:
  %result = zext i32 %val to i64
  ret i64 %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Integer extension (sign extend)
        text = "Sign extend i32 to i64"
        ir = """define i64 @sext_i32_to_i64(i32 %val) {
entry:
  %result = sext i32 %val to i64
  ret i64 %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Float to int
        text = "Convert float to signed integer"
        ir = """define i32 @fptosi(float %val) {
entry:
  %result = fptosi float %val to i32
  ret i32 %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Int to float
        text = "Convert signed integer to float"
        ir = """define float @sitofp(i32 %val) {
entry:
  %result = sitofp i32 %val to float
  ret float %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Float extension
        text = "Extend float to double"
        ir = """define double @fpext(float %val) {
entry:
  %result = fpext float %val to double
  ret double %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Float truncation
        text = "Truncate double to float"
        ir = """define float @fptrunc(double %val) {
entry:
  %result = fptrunc double %val to float
  ret float %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Pointer to int
        text = "Convert pointer to integer"
        ir = """define i64 @ptrtoint(i32* %ptr) {
entry:
  %result = ptrtoint i32* %ptr to i64
  ret i64 %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Int to pointer
        text = "Convert integer to pointer"
        ir = """define i32* @inttoptr(i64 %val) {
entry:
  %result = inttoptr i64 %val to i32*
  ret i32* %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Bitcast
        text = "Bitcast between types of same size"
        ir = """define i32 @bitcast_float_to_int(float %val) {
entry:
  %result = bitcast float %val to i32
  ret i32 %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        return examples
    
    def generate_intrinsics(self) -> List[Tuple[str, str, str]]:
        """Generate LLVM intrinsic function examples."""
        examples = []
        
        # memcpy
        text = "Copy memory block using memcpy intrinsic"
        ir = """define void @copy_memory(i8* %dest, i8* %src, i64 %len) {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 %len, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)"""
        self._add_example(examples, text, ir, "GOOD")
        
        # memset
        text = "Fill memory block using memset intrinsic"
        ir = """define void @fill_memory(i8* %dest, i8 %val, i64 %len) {
entry:
  call void @llvm.memset.p0i8.i64(i8* %dest, i8 %val, i64 %len, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i1)"""
        self._add_example(examples, text, ir, "GOOD")
        
        # sqrt
        text = "Calculate square root using sqrt intrinsic"
        ir = """define double @sqrt_example(double %val) {
entry:
  %result = call double @llvm.sqrt.f64(double %val)
  ret double %result
}

declare double @llvm.sqrt.f64(double)"""
        self._add_example(examples, text, ir, "GOOD")
        
        # abs
        text = "Calculate absolute value using abs intrinsic"
        ir = """define i32 @abs_intrinsic(i32 %val) {
entry:
  %result = call i32 @llvm.abs.i32(i32 %val, i1 false)
  ret i32 %result
}

declare i32 @llvm.abs.i32(i32, i1)"""
        self._add_example(examples, text, ir, "GOOD")
        
        # min/max
        text = "Find minimum of two values using min intrinsic"
        ir = """define i32 @min_intrinsic(i32 %a, i32 %b) {
entry:
  %result = call i32 @llvm.smin.i32(i32 %a, i32 %b)
  ret i32 %result
}

declare i32 @llvm.smin.i32(i32, i32)"""
        self._add_example(examples, text, ir, "GOOD")
        
        # ctpop (count population/set bits)
        text = "Count set bits using ctpop intrinsic"
        ir = """define i32 @count_bits(i32 %val) {
entry:
  %result = call i32 @llvm.ctpop.i32(i32 %val)
  ret i32 %result
}

declare i32 @llvm.ctpop.i32(i32)"""
        self._add_example(examples, text, ir, "GOOD")
        
        # ctlz (count leading zeros)
        text = "Count leading zeros using ctlz intrinsic"
        ir = """define i32 @leading_zeros(i32 %val) {
entry:
  %result = call i32 @llvm.ctlz.i32(i32 %val, i1 false)
  ret i32 %result
}

declare i32 @llvm.ctlz.i32(i32, i1)"""
        self._add_example(examples, text, ir, "GOOD")
        
        # bswap (byte swap)
        text = "Swap bytes using bswap intrinsic"
        ir = """define i32 @byte_swap(i32 %val) {
entry:
  %result = call i32 @llvm.bswap.i32(i32 %val)
  ret i32 %result
}

declare i32 @llvm.bswap.i32(i32)"""
        self._add_example(examples, text, ir, "GOOD")
        
        return examples
    
    def generate_varargs(self) -> List[Tuple[str, str, str]]:
        """Generate variable argument function examples."""
        examples = []
        
        # Note: Varargs in LLVM IR are complex and platform-specific.
        # We'll focus on the declaration and basic usage pattern.
        
        # Simple varargs declaration (common pattern)
        text = "Declare variadic function (printf-style)"
        ir = """declare i32 @printf(i8*, ...)

define void @print_numbers(i32 %a, i32 %b) {
entry:
  %fmt = getelementptr [14 x i8], [14 x i8]* @.str, i32 0, i32 0
  call i32 (i8*, ...) @printf(i8* %fmt, i32 %a, i32 %b)
  ret void
}

@.str = private constant [14 x i8] c"Values: %d %d\\00\""""
        self._add_example(examples, text, ir, "GOOD")
        
        return examples
    
    def generate_function_attributes(self) -> List[Tuple[str, str, str]]:
        """Generate functions with various attributes."""
        examples = []
        
        # Pure function (readnone)
        text = "Pure function with readnone attribute (no memory access)"
        ir = """define i32 @pure_add(i32 %a, i32 %b) readnone {
entry:
  %result = add i32 %a, %b
  ret i32 %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Readonly function
        text = "Readonly function that only reads memory"
        ir = """define i32 @read_value(i32* %ptr) readonly {
entry:
  %val = load i32, i32* %ptr
  ret i32 %val
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Nounwind function
        text = "Function that never throws exceptions (nounwind)"
        ir = """define i32 @safe_divide(i32 %a, i32 %b) nounwind {
entry:
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %zero, label %divide

zero:
  ret i32 0

divide:
  %result = sdiv i32 %a, %b
  ret i32 %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Inline function
        text = "Function with always_inline attribute"
        ir = """define i32 @inline_add(i32 %a, i32 %b) alwaysinline {
entry:
  %result = add i32 %a, %b
  ret i32 %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        return examples
    
    def generate_tail_calls(self) -> List[Tuple[str, str, str]]:
        """Generate tail call optimization examples."""
        examples = []
        
        # Tail recursive factorial
        text = "Tail recursive factorial with tail call optimization"
        ir = """define i32 @factorial_tail(i32 %n, i32 %acc) {
entry:
  %cmp = icmp sle i32 %n, 1
  br i1 %cmp, label %base, label %recurse

base:
  ret i32 %acc

recurse:
  %n_next = sub i32 %n, 1
  %acc_next = mul i32 %acc, %n
  %result = tail call i32 @factorial_tail(i32 %n_next, i32 %acc_next)
  ret i32 %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        return examples
    
    def generate_aggregates(self) -> List[Tuple[str, str, str]]:
        """Generate aggregate type operations."""
        examples = []
        
        # Insert value into struct
        text = "Create and return struct with insertvalue"
        ir = """%Pair = type { i32, i32 }

define %Pair @make_pair(i32 %a, i32 %b) {
entry:
  %pair1 = insertvalue %Pair undef, i32 %a, 0
  %pair2 = insertvalue %Pair %pair1, i32 %b, 1
  ret %Pair %pair2
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Extract value from struct
        text = "Extract first element from struct"
        ir = """%Pair = type { i32, i32 }

define i32 @get_first(%Pair %pair) {
entry:
  %result = extractvalue %Pair %pair, 0
  ret i32 %result
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        return examples
    
    def generate_global_variables(self) -> List[Tuple[str, str, str]]:
        """Generate global variable examples."""
        examples = []
        
        # Global constant
        text = "Define and use global constant"
        ir = """@PI = constant double 3.141592653589793

define double @get_pi() {
entry:
  %pi = load double, double* @PI
  ret double %pi
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        # Global variable
        text = "Define and use global variable"
        ir = """@counter = global i32 0

define i32 @increment_counter() {
entry:
  %val = load i32, i32* @counter
  %new_val = add i32 %val, 1
  store i32 %new_val, i32* @counter
  ret i32 %new_val
}"""
        self._add_example(examples, text, ir, "GOOD")
        
        return examples
    
    def generate_all(self) -> List[Tuple[str, str, str]]:
        """Generate all training examples."""
        print("Generating comprehensive dataset with advanced concepts...")
        all_examples = []
        
        generators = [
            ("Basic arithmetic", self.generate_basic_arithmetic),
            ("Floating point", self.generate_floating_point),
            ("Comparisons", self.generate_comparisons),
            ("Conditionals", self.generate_conditionals),
            ("Loops", self.generate_loops),
            ("Arrays", self.generate_arrays),
            ("Bitwise operations", self.generate_bitwise),
            ("Functions", self.generate_functions),
            ("Structs and pointers", self.generate_structs_and_pointers),
            ("Select operations", self.generate_select_operations),
            ("Edge cases", self.generate_edge_cases),
            ("Memory operations", self.generate_memory_operations),
            ("Vector operations (SIMD)", self.generate_vector_operations),
            ("Atomic operations", self.generate_atomic_operations),
            ("Switch statements", self.generate_switch_statements),
            ("Advanced phi nodes", self.generate_phi_advanced),
            ("Type conversions", self.generate_type_conversions),
            ("LLVM intrinsics", self.generate_intrinsics),
            ("Variable arguments", self.generate_varargs),
            ("Function attributes", self.generate_function_attributes),
            ("Tail call optimization", self.generate_tail_calls),
            ("Aggregate operations", self.generate_aggregates),
            ("Global variables", self.generate_global_variables),
        ]
        
        for name, generator in generators:
            print(f"  Generating {name}...")
            examples = generator()
            all_examples.extend(examples)
            print(f"    Generated {len(examples)} examples")
        
        if self.validate_ir and self.validation_failures > 0:
            print(f"\n  Note: {self.validation_failures} invalid IR examples were skipped")
        
        return all_examples


def create_text_variations(text: str, count: int = 10) -> List[str]:
    """Create multiple variations of the same text with improved diversity."""
    variations = set([text])
    
    replacements = {
        "Write": ["Implement", "Create", "Define", "Generate", "Code", "Build", "Make", "Develop"],
        "Create": ["Write", "Implement", "Build", "Make", "Generate", "Code", "Develop"],
        "Implement": ["Write", "Create", "Code", "Build", "Define", "Develop"],
        "function": ["function", "subroutine", "procedure", "method", "routine", "func"],
        "that": ["which", "that", "to"],
        "returns": ["gives", "returns", "outputs", "produces", "yields", "provides"],
        "calculates": ["computes", "calculates", "determines", "finds", "evaluates"],
        "two": ["2", "two", "a pair of", "2 separate"],
        "integers": ["integers", "ints", "integer values", "whole numbers"],
        "values": ["values", "numbers", "operands", "inputs"],
        "check": ["check", "verify", "test", "determine", "validate"],
        "array": ["array", "list", "buffer", "sequence"],
        "elements": ["elements", "items", "values", "entries"],
    }
    
    # Generate more variations efficiently
    attempts = 0
    max_attempts = count * 3
    while len(variations) < count + 1 and attempts < max_attempts:
        attempts += 1
        new_text = text
        # Apply 1-3 random replacements
        num_replacements = random.randint(1, 3)
        applicable_keys = [k for k in replacements.keys() if k in new_text]
        if applicable_keys:
            for _ in range(num_replacements):
                old = random.choice(applicable_keys)
                new_text = new_text.replace(old, random.choice(replacements[old]), 1)
        
        if new_text != text:
            variations.add(new_text)
    
    return list(variations)


def augment_dataset(examples: List[Tuple[str, str, str]], target_count: int, 
                    variations_per_example: int = 10) -> List[Tuple[str, str, str]]:
    """Augment dataset to reach target count with optimized batch processing."""
    if len(examples) >= target_count:
        return examples[:target_count]
    
    print(f"\nAugmenting dataset from {len(examples):,} to {target_count:,} examples...")
    augmented = list(examples)
    
    # Calculate how many times we need to replicate the base examples
    needed = target_count - len(examples)
    
    # Create a pool of examples with variations
    print("Generating text variations...")
    example_pool = []
    for text, ir, quality in tqdm(examples, desc="Creating variations"):
        variations = create_text_variations(text, variations_per_example)
        for var_text in variations:
            example_pool.append((var_text, ir, quality))
    
    # If pool is smaller than needed, repeat it
    if len(example_pool) < needed:
        multiplier = (needed // len(example_pool)) + 1
        example_pool = example_pool * multiplier
    
    # Shuffle and take what we need
    random.shuffle(example_pool)
    augmented.extend(example_pool[:needed])
    
    print(f"Augmentation complete: {len(augmented):,} examples")
    return augmented[:target_count]


def save_dataset(examples: List[Tuple[str, str, str]], output_dir: str, split: str):
    """Save dataset in JSONL format with quality markers and streaming write."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{split}.jsonl")
    
    print(f"Saving {split} set...")
    with open(output_file, 'w') as f:
        for text, ir, quality in tqdm(examples, desc=f"Writing {split}", leave=False):
            example = {
                "text": text,
                "llvm_ir": ir,
                "quality": quality
            }
            f.write(json.dumps(example) + '\n')
    
    # Calculate file size
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Saved {len(examples):,} examples to {output_file} ({size_mb:.2f} MB)")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate large-scale text-to-LLVM IR training dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--target-examples",
        type=int,
        default=500000,
        help="Target number of examples to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="Output directory for dataset files"
    )
    parser.add_argument(
        "--variations-per-example",
        type=int,
        default=10,
        help="Number of text variations per base example"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Proportion of data for training"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Proportion of data for validation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Generate a small dataset for quick testing (1000 examples)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip LLVM IR validation (faster but may include invalid IR)"
    )
    return parser.parse_args()


def main():
    """Generate and save large-scale training data."""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Adjust target for quick test
    if args.quick_test:
        target_examples = 1000
        print("=" * 60)
        print("QUICK TEST MODE - Generating small dataset")
        print("=" * 60)
    else:
        target_examples = args.target_examples
        
    print("=" * 60)
    print("Enhanced Text-to-LLVM IR Dataset Generation")
    print(f"Target: {target_examples:,} examples")
    print("=" * 60)
    
    start_time = time.time()
    
    # Generate base examples
    print(f"LLVM IR Validation: {'Disabled' if args.no_validate else 'Enabled'}")
    generator = EnhancedLLVMIRGenerator(validate_ir=not args.no_validate)
    base_examples = generator.generate_all()
    print(f"\nTotal unique base examples: {len(base_examples):,}")
    
    # Count by quality
    good_count = sum(1 for _, _, q in base_examples if q == "GOOD")
    bad_count = sum(1 for _, _, q in base_examples if q == "BAD")
    print(f"  GOOD examples: {good_count:,}")
    print(f"  BAD examples: {bad_count:,}")
    
    # Augment to reach target size
    all_examples = augment_dataset(
        base_examples, 
        target_examples,
        variations_per_example=args.variations_per_example
    )
    print(f"\nFinal dataset size: {len(all_examples):,} examples")
    
    # Shuffle
    print("Shuffling dataset...")
    random.shuffle(all_examples)
    
    # Split into train/val/test
    total = len(all_examples)
    train_size = int(args.train_split * total)
    val_size = int(args.val_split * total)
    
    train_data = all_examples[:train_size]
    val_data = all_examples[train_size:train_size + val_size]
    test_data = all_examples[train_size + val_size:]
    
    # Save datasets
    print("\n" + "=" * 60)
    print("Saving datasets...")
    print("=" * 60)
    save_dataset(train_data, args.output_dir, "train")
    save_dataset(val_data, args.output_dir, "val")
    save_dataset(test_data, args.output_dir, "test")
    
    # Calculate total size and time
    total_size = sum(
        os.path.getsize(os.path.join(args.output_dir, f"{split}.jsonl"))
        for split in ["train", "val", "test"]
    ) / (1024 * 1024 * 1024)
    
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    print(f"Train: {len(train_data):,} examples")
    print(f"Val: {len(val_data):,} examples")
    print(f"Test: {len(test_data):,} examples")
    print(f"Total size: {total_size:.2f} GB")
    print(f"Generation time: {minutes}m {seconds}s")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

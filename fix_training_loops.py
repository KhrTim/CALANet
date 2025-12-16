#!/usr/bin/env python3
"""
Script to wrap training loops with metrics_collector.track_training()
for all remaining experiment scripts.
"""

import os
import re

# Files that need training loop wrapping
FILES_TO_FIX = [
    'codes/RevTransformerAttentionHAR/run.py',
    'codes/DeepConvLSTM/run.py',
    'codes/resnet/run_TSC.py',
    'codes/InceptionTime/run_TSC.py',
    'codes/DSN-master/run.py',
    'codes/DSN-master/run_TSC.py',
    'codes/FCN_TSC/run_TSC.py',
    'codes/IF-ConvTransformer2/run.py',
    'codes/RepHAR/run.py',
    'codes/Bi-GRU-I/run.py',
]

def wrap_training_loop(content):
    """
    Wrap training loop with metrics collector tracking.
    Removes TODO comments and adds proper indentation.
    """
    # Pattern to match the TODO section and the for epoch line
    pattern = r'(# TODO: Wrap training loop.*?\n)((?:#.*?\n)*)\n(for epoch in range\(epoches\):)'

    def replacer(match):
        # Just return the wrapped version
        return '# Track training time\nwith metrics_collector.track_training():\n    for epoch in range(epoches):'

    # Replace the TODO with wrapped version
    content = re.sub(pattern, replacer, content, flags=re.DOTALL)

    # Now we need to indent everything inside the for loop
    # This is tricky because we need to find where the loop ends
    # The loop typically ends when we reach a line that's not indented (like "print")

    # Split into lines for easier processing
    lines = content.split('\n')

    # Find the line with "with metrics_collector.track_training():"
    in_training_context = False
    in_for_loop = False
    result_lines = []

    for i, line in enumerate(lines):
        if 'with metrics_collector.track_training():' in line and 'metrics_collector.track_training' in line:
            in_training_context = True
            result_lines.append(line)
        elif in_training_context and 'for epoch in range(epoches):' in line:
            in_for_loop = True
            result_lines.append(line)
        elif in_for_loop:
            # Check if this is the end of the for loop
            # Loop ends when we encounter a line that starts with no indentation or only 0-3 spaces
            # and is not empty
            stripped = line.lstrip()
            if stripped and not line.startswith('    '):
                # End of loop
                in_for_loop = False
                in_training_context = False
                result_lines.append(line)
            else:
                # Inside loop - add 4 spaces of indentation
                if line.strip():  # Not empty
                    result_lines.append('    ' + line)
                else:
                    result_lines.append(line)
        else:
            result_lines.append(line)

    return '\n'.join(result_lines)

def process_file(filepath):
    """Process a single file to wrap its training loop."""
    print(f"Processing {filepath}...")

    # Read the file
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if it has the TODO
    if 'TODO: Wrap training loop' not in content:
        print(f"  Skipping - no TODO found")
        return False

    # Backup the file
    backup_path = filepath + '.before_training_wrap'
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"  Created backup: {backup_path}")

    # Wrap the training loop
    new_content = wrap_training_loop(content)

    # Write back
    with open(filepath, 'w') as f:
        f.write(new_content)

    print(f"  ✓ Wrapped training loop")
    return True

def main():
    """Main function."""
    print("="*70)
    print("Wrapping Training Loops with Metrics Collector")
    print("="*70)
    print()

    fixed_count = 0
    for filepath in FILES_TO_FIX:
        if os.path.exists(filepath):
            if process_file(filepath):
                fixed_count += 1
        else:
            print(f"File not found: {filepath}")

    print()
    print("="*70)
    print(f"COMPLETE: Fixed {fixed_count} files")
    print("="*70)

if __name__ == '__main__':
    main()

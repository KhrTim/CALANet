#!/usr/bin/env python3
"""
Fix metrics collector placement that's breaking optimizer definitions
"""

import os
import re

FILES_TO_FIX = [
    'codes/RepHAR/run.py',
    'codes/DeepConvLSTM/run.py',
    'codes/Bi-GRU-I/run.py',
    'codes/RevTransformerAttentionHAR/run.py',
    'codes/IF-ConvTransformer2/run.py',
]

def fix_file(filepath):
    """Fix the metrics collector placement in a file."""
    print(f"Fixing {filepath}...")

    with open(filepath, 'r') as f:
        content = f.read()

    # Backup
    with open(filepath + '.before_optimizer_fix', 'w') as f:
        f.write(content)

    # Pattern to find broken optimizer + metrics collector
    # Look for: optimizer = torch.optim.Adam(
    #
    #           # Initialize metrics collector
    #           metrics_collector = MetricsCollector(
    #               ...
    #           )
    #           model.parameters() or similar...

    lines = content.split('\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this is the optimizer line
        if 'optimizer = torch.optim.Adam(' in line:
            # Collect the optimizer definition line
            optimizer_line = line
            fixed_lines.append(optimizer_line)
            i += 1

            # Check if next few lines have the metrics collector
            metrics_start = None
            for j in range(i, min(i + 3, len(lines))):
                if 'metrics_collector = MetricsCollector(' in lines[j]:
                    metrics_start = j
                    break

            if metrics_start is not None:
                # Found the broken pattern
                # Skip empty lines and comment
                while i < metrics_start:
                    if lines[i].strip() and not lines[i].strip().startswith('#'):
                        fixed_lines.append(lines[i])
                    i += 1

                # Collect the entire metrics_collector block
                metrics_lines = []
                paren_count = 0
                j = metrics_start
                while j < len(lines):
                    metrics_lines.append(lines[j])
                    paren_count += lines[j].count('(') - lines[j].count(')')
                    j += 1
                    if paren_count <= 0 and 'metrics_collector = MetricsCollector(' in ''.join(metrics_lines):
                        break

                # Now find the rest of the optimizer definition
                optimizer_rest_lines = []
                while j < len(lines):
                    if lines[j].strip().startswith(')') or 'model.parameters()' in lines[j] or 'parameters.contiguous()' in lines[j]:
                        # Found the end of optimizer
                        # Collect until we hit the closing )
                        while j < len(lines):
                            optimizer_rest_lines.append(lines[j])
                            if ')' in lines[j] and not lines[j].strip().startswith('#'):
                                j += 1
                                break
                            j += 1
                        break
                    optimizer_rest_lines.append(lines[j])
                    j += 1

                # Now reconstruct properly: optimizer first, then metrics
                # Add the rest of optimizer definition
                for ol in optimizer_rest_lines:
                    fixed_lines.append(ol)

                # Add empty line
                fixed_lines.append('')

                # Add metrics collector
                for ml in metrics_lines:
                    fixed_lines.append(ml)

                i = j
            else:
                # No issue found, continue normally
                continue
        else:
            fixed_lines.append(line)
            i += 1

    # Write fixed content
    with open(filepath, 'w') as f:
        f.write('\n'.join(fixed_lines))

    print(f"  ✓ Fixed {filepath}")

def main():
    print("="*70)
    print("Fixing Optimizer + MetricsCollector Placements")
    print("="*70)

    for filepath in FILES_TO_FIX:
        if os.path.exists(filepath):
            fix_file(filepath)
        else:
            print(f"  File not found: {filepath}")

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)

if __name__ == '__main__':
    main()

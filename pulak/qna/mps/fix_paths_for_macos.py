#!/usr/bin/env python3
"""
Script to fix hardcoded Linux paths in MISO codebase for macOS client setup.
This makes the code work with relative paths instead of hardcoded /home/USER/GIT/socc22-miso
"""

import os
import re
import sys
from pathlib import Path

# Get repository root (assuming script is in pulak/qna/mps/)
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent.absolute()

print(f"Repository root: {REPO_ROOT}")

# Files that need path fixes
files_to_fix = [
    'exp_mps.py',
    'controller_helper.py',
    'exp_full.py',
    'exp_oracle.py',
    'exp_static.py',
    'exp_miso.py',
    'export_cuda_device_auto.py',
    'dummy/dummy_sender.py',
]

def fix_paths_in_file(filepath):
    """Replace hardcoded paths with relative paths"""
    if not os.path.exists(filepath):
        print(f"  ⚠️  File not found: {filepath}")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: sys.path.append(f'/home/{user}/GIT/socc22-miso/...')
    content = re.sub(
        r"sys\.path\.append\(f'/home/\{user\}/GIT/socc22-miso/([^']+)'\)",
        lambda m: f"sys.path.append(str(Path(__file__).parent.parent / '{m.group(1)}'))",
        content
    )
    
    # Pattern 2: open(f'/home/{user}/GIT/socc22-miso/...')
    content = re.sub(
        r"open\(f'/home/\{user\}/GIT/socc22-miso/([^']+)'\)",
        lambda m: f"open(Path(__file__).parent.parent / '{m.group(1)}')",
        content
    )
    
    # Pattern 3: f'/home/{user}/GIT/socc22-miso/...' in strings
    content = re.sub(
        r"f'/home/\{user\}/GIT/socc22-miso/([^']+)'",
        lambda m: f"str(Path(__file__).parent.parent / '{m.group(1)}')",
        content
    )
    
    # Add Path import if we made changes and it's not already there
    if content != original_content:
        if 'from pathlib import Path' not in content and 'import Path' not in content:
            # Find the last import statement
            import_pattern = r'(import\s+\w+|from\s+\w+\s+import)'
            imports = list(re.finditer(import_pattern, content))
            if imports:
                last_import = imports[-1]
                insert_pos = last_import.end()
                content = content[:insert_pos] + '\nfrom pathlib import Path' + content[insert_pos:]
            else:
                # Add at the top after shebang
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        lines.insert(i, 'from pathlib import Path')
                        break
                content = '\n'.join(lines)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✅ Fixed: {filepath}")
        return True
    else:
        print(f"  ℹ️  No changes needed: {filepath}")
        return False

def main():
    print("Fixing hardcoded paths for macOS client setup...")
    print("=" * 60)
    
    # Change to repo root
    os.chdir(REPO_ROOT)
    
    fixed_count = 0
    for filename in files_to_fix:
        filepath = REPO_ROOT / filename
        if fix_paths_in_file(filepath):
            fixed_count += 1
    
    print("=" * 60)
    print(f"✅ Fixed {fixed_count} files")
    print("\n⚠️  Note: You may need to manually review and fix:")
    print("   - gpu_server.py (has many hardcoded paths)")
    print("   - Any files that reference /scratch/${USER}")
    print("   - Physical node hostnames in run scripts")
    print("\nSee macos_client_setup.md for full instructions.")

if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Convert an existing Python geospatial training directory structure to a MkDocs site.
Automatically discovers day READMEs and prepares a lean documentation site.
"""

from pathlib import Path
import yaml

class MkDocsConverter:
    def __init__(self, src_dir="src", output_dir="docs"):
        self.src_dir = Path(src_dir)
        self.output_dir = Path(output_dir)
        self.docs_dir = self.output_dir / "docs"
        
    def setup_mkdocs_structure(self):
        """Create the MkDocs directory structure."""
        print("🏗️  Setting up MkDocs structure...")
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.docs_dir.mkdir(exist_ok=True)
        (self.docs_dir / "stylesheets").mkdir(exist_ok=True)
        (self.docs_dir / "javascripts").mkdir(exist_ok=True)
        (self.docs_dir / "code").mkdir(exist_ok=True)
        
    def create_mkdocs_config(self):
        """Generate a lean mkdocs.yml configuration."""
        config = {
            'site_name': 'Python Geospatial Engineering Practices',
            'site_description': 'A practical prep curriculum for senior Python engineers',
            'theme': {
                'name': 'material',
                'palette': [
                    {'scheme': 'default', 'primary': 'blue grey', 'accent': 'red'},
                    {'scheme': 'slate', 'primary': 'blue grey', 'accent': 'red'},
                ],
                'features': [
                    'navigation.tabs',
                    'navigation.sections',
                    'toc.follow',
                    'search.suggest',
                    'content.code.copy',
                ],
            },
            'plugins': [
                'search',
                {'mkdocstrings': {
                    'handlers': {
                        'python': {
                            'setup_commands': [
                                'import sys, os',
                                'sys.path.insert(0, os.path.abspath(".."))',
                            ],
                            'options': {
                                'show_source': True,
                                'show_root_heading': True,
                            },
                        }
                    }
                }},
            ],
            'markdown_extensions': [
                'admonition',
                'tables',
                'toc',
                'attr_list',
                'pymdownx.details',
                'pymdownx.highlight',
                'pymdownx.superfences',
                'pymdownx.tabbed',
                'pymdownx.emoji',
                'pymdownx.magiclink',
            ],
            'extra_css': ['stylesheets/extra.css'],
            'extra_javascript': ['javascripts/config.js'],
        }
        
        # Build navigation from directory structure
        config['nav'] = self.build_navigation()
        
        # Write config file
        config_path = self.output_dir / "mkdocs.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Created {config_path}")
        
    def build_navigation(self):
        """Build navigation from src directory (only generated files)."""
        nav = [
            {'Home': 'index.md'},
            {'Overview': [
                {'Study Plan': 'study_plan.md'},
            ]},
        ]

        days = []
        for day_dir in sorted(self.src_dir.glob("day*")):
            if day_dir.is_dir():
                day_nav = self.process_day_directory(day_dir)
                if day_nav:
                    days.append(day_nav)

        if days:
            nav.append({'Training Modules': days})

        # API docs - temporarily disabled due to path issues
        # api_items = []
        # if (self.docs_dir / 'api' / 'day01_tile_fetcher.md').exists():
        #     api_items.append({'Day 1 Tile Fetcher': 'api/day01_tile_fetcher.md'})
        # if (self.docs_dir / 'api' / 'day03_api.md').exists():
        #     nav.append({'API': api_items})

        return nav
    
    def process_day_directory(self, day_dir):
        """Process a single day directory."""
        day_name = day_dir.name
        day_num = day_name.split('_')[0].replace('day', 'Day ')
        topic = ' '.join(day_name.split('_')[1:]).title()
        
        day_title = f"{day_num} - {topic}"
        day_items = []
        
        # Copy README if exists
        readme_src = day_dir / "README.md"
        if readme_src.exists():
            readme_dest = self.docs_dir / day_name / "index.md"
            self.copy_and_enhance_readme(readme_src, readme_dest, day_title)
            day_items.append({'Overview': f"{day_name}/index.md"})
        
        # Keep nav minimal — README per day is enough for learning flow.
        
        if day_items:
            return {day_title: day_items}
        return None
    
    def process_subdirectory(self, subdir, parent_name):
        """(Unused in lean build)."""
        return []
    
    def copy_and_enhance_readme(self, src_path, dest_path, title):
        """Copy README to docs with a simple title wrapper and clean up redundant headers."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(src_path, 'r') as f:
            content = f.read()
        
        # Clean up redundant H1 headers - keep only the most descriptive one
        lines = content.split('\n')
        cleaned_lines = []
        h1_count = 0
        
        for line in lines:
            if line.startswith('# '):
                h1_count += 1
                if h1_count == 1:
                    # Skip the first H1 (usually just "Day X - Topic")
                    continue
                else:
                    # Keep subsequent H1s (the more descriptive ones)
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        
        # If no H1s were found, use the original content
        if h1_count == 0:
            cleaned_content = content
        else:
            cleaned_content = '\n'.join(cleaned_lines).strip()
        
        enhanced_content = f"# {title}\n\n{cleaned_content}\n"
        with open(dest_path, 'w') as f:
            f.write(enhanced_content)
    
    def create_code_documentation(self, py_file, parent_path):
        """Create documentation page for Python file."""
        dest_dir = self.docs_dir / parent_path
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        doc_filename = f"{py_file.stem}.md"
        doc_path = dest_dir / doc_filename
        
        with open(py_file, 'r') as f:
            code = f.read()
        
        # Extract docstrings and create documentation
        doc_content = f"""# {py_file.stem.replace('_', ' ').title()}

## Source Code

```python
{code}
```

## Key Concepts

This module demonstrates:
{self.extract_key_concepts(code)}

## Usage Example

```python
# Import and use the module
from {parent_path.replace('/', '.')}.{py_file.stem} import *

# Example usage would go here
```

## Testing

Run tests with:
```bash
pytest tests/test_{py_file.stem}.py -v --cov={py_file.stem}
```

## Performance Notes

!!! tip "Optimization"
    Consider profiling this code with `cProfile` to identify bottlenecks.

"""
        
        with open(doc_path, 'w') as f:
            f.write(doc_content)
        
        return f"{parent_path}/{doc_filename}"
    
    def extract_key_concepts(self, code):
        """Extract key concepts from Python code."""
        concepts = []
        
        if 'async def' in code:
            concepts.append("- **Async/Await** patterns for concurrent operations")
        if 'class ' in code:
            concepts.append("- **Object-Oriented Design** with classes and methods")
        if '@' in code and 'def' in code:
            concepts.append("- **Decorators** for enhanced functionality")
        if 'ThreadPoolExecutor' in code or 'ProcessPoolExecutor' in code:
            concepts.append("- **Concurrent execution** with thread/process pools")
        if 'pytest' in code or 'unittest' in code:
            concepts.append("- **Unit testing** practices")
        if 'FastAPI' in code or 'APIRouter' in code:
            concepts.append("- **REST API** development with FastAPI")
        
        return '\n'.join(concepts) if concepts else "- Review the code to identify key patterns"
    
    def create_data_documentation(self, data_file, parent_path):
        """Create documentation for data files."""
        dest_dir = self.docs_dir / parent_path
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        doc_filename = f"data_{data_file.stem}.md"
        doc_path = dest_dir / doc_filename
        
        # Read first few lines of data
        with open(data_file, 'r') as f:
            lines = f.readlines()[:10]
        
        doc_content = f"""# Data: {data_file.name}

## Sample Data

```csv
{''.join(lines)}
```

## Schema Description

Analyze the data structure and document the schema here.

## Usage in Code

```python
import pandas as pd

# Load the data
df = pd.read_csv('{data_file.name}')

# Basic exploration
print(df.head())
print(df.info())
print(df.describe())
```
"""
        
        with open(doc_path, 'w') as f:
            f.write(doc_content)
        
        return f"{parent_path}/{doc_filename}"
    
    def create_proto_documentation(self, proto_file, parent_path):
        """Create documentation for Protocol Buffer files."""
        dest_dir = self.docs_dir / parent_path
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        doc_filename = f"proto_{proto_file.stem}.md"
        doc_path = dest_dir / doc_filename
        
        with open(proto_file, 'r') as f:
            proto_content = f.read()
        
        doc_content = f"""# Protocol Buffer: {proto_file.name}

## Definition

```protobuf
{proto_content}
```

## Python Usage

```python
# Compile the proto file first:
# protoc --python_out=. {proto_file.name}

import {proto_file.stem}_pb2

# Create a message
msg = {proto_file.stem}_pb2.RoadSegment()
msg.id = "R001"
msg.name = "El Camino Real"

# Serialize
data = msg.SerializeToString()

# Deserialize
new_msg = {proto_file.stem}_pb2.RoadSegment()
new_msg.ParseFromString(data)
```

## Integration with gRPC

```python
# For gRPC service integration
import grpc
import {proto_file.stem}_pb2
import {proto_file.stem}_pb2_grpc

# Implementation would go here
```
"""
        
        with open(doc_path, 'w') as f:
            f.write(doc_content)
        
        return f"{parent_path}/{doc_filename}"
    
    def create_index_page(self):
        """Create the main index page from repo README and a Study Plan from PLAN.md."""
        index_path = self.docs_dir / "index.md"
        readme_root = Path("README.md")
        if readme_root.exists():
            with open(readme_root, 'r') as fsrc, open(index_path, 'w') as fdst:
                fdst.write(fsrc.read())
        else:
            with open(index_path, 'w') as fdst:
                fdst.write("# Python Geospatial Engineering Practices\n\nWelcome.")

        print(f"✅ Created {index_path}")

        plan_src = Path("docs/PLAN.md")
        if plan_src.exists():
            plan_dest = self.docs_dir / "study_plan.md"
            with open(plan_src, 'r') as fsrc, open(plan_dest, 'w') as fdst:
                fdst.write(fsrc.read())
            print(f"✅ Created {plan_dest}")
        else:
            # Fallback if PLAN.md doesn't exist
            plan_dest = self.docs_dir / "study_plan.md"
            with open(plan_dest, 'w') as fdst:
                fdst.write("# Study Plan\n\nStudy plan content will be generated here.")
            print(f"✅ Created fallback {plan_dest}")
    
    def create_extra_files(self):
        """Create CSS and JS files for enhanced styling."""
        
        # CSS file
        css_content = """
/* Brand-inspired colors */
:root {
    --md-primary-fg-color: #1e3a8a;
    --md-accent-fg-color: #dc2626;
    --brand-blue: #0050ff;
    --brand-red: #ff0000;
}

/* Progress bar */
.progress-bar {
    width: 100%;
    height: 25px;
    background: #e5e7eb;
    border-radius: 12px;
    overflow: hidden;
    margin: 20px 0;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--brand-blue), var(--brand-red));
    transition: width 0.5s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
}

/* Code blocks */
.highlight pre {
    border-left: 4px solid var(--md-accent-fg-color);
    border-radius: 4px;
}

/* Cards grid */
.grid.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.grid.cards > * {
    padding: 1rem;
    border: 1px solid var(--md-default-fg-color--lightest);
    border-radius: 8px;
    transition: transform 0.2s;
}

.grid.cards > *:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* Animation for checkboxes */
.task-list-control:checked {
    animation: checkmark 0.3s ease;
}

@keyframes checkmark {
    0% { transform: scale(1); }
    50% { transform: scale(1.2); }
    100% { transform: scale(1); }
}
"""
        
        css_path = self.docs_dir / "stylesheets" / "extra.css"
        css_path.parent.mkdir(parents=True, exist_ok=True)
        with open(css_path, 'w') as f:
            f.write(css_content)
        
        # JavaScript file
        js_content = """
// Progress tracking
document.addEventListener('DOMContentLoaded', function() {
    // Count checked boxes
    const checkboxes = document.querySelectorAll('.task-list-control');
    const progressBar = document.querySelector('.progress-fill');
    
    function updateProgress() {
        const total = checkboxes.length;
        const checked = document.querySelectorAll('.task-list-control:checked').length;
        const percentage = (checked / total) * 100;
        
        if (progressBar) {
            progressBar.style.width = percentage + '%';
            progressBar.textContent = Math.round(percentage) + '%';
        }
        
        // Save to localStorage
        localStorage.setItem('python-prep-progress', percentage);
    }
    
    // Load saved progress
    const savedProgress = localStorage.getItem('python-prep-progress');
    if (savedProgress && progressBar) {
        progressBar.style.width = savedProgress + '%';
        progressBar.textContent = Math.round(savedProgress) + '%';
    }
    
    // Listen for changes
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', updateProgress);
    });
    
    updateProgress();
});

// Code copy enhancement
document.addEventListener('DOMContentLoaded', function() {
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        block.addEventListener('click', function() {
            // Flash effect on copy
            block.style.background = '#e3f2fd';
            setTimeout(() => {
                block.style.background = '';
            }, 200);
        });
    });
});
"""
        
        js_path = self.docs_dir / "javascripts" / "config.js"
        js_path.parent.mkdir(parents=True, exist_ok=True)
        with open(js_path, 'w') as f:
            f.write(js_content)
        
        print(f"✅ Created styling files")
    
    def create_requirements_txt(self):
        """Create requirements.txt for MkDocs (lean)."""
        requirements = """mkdocs>=1.5.0
mkdocs-material>=9.0.0
mkdocstrings>=0.24.0
mkdocstrings-python>=1.7.0
pymdown-extensions>=10.0
"""
        
        req_path = self.output_dir / "requirements.txt"
        with open(req_path, 'w') as f:
            f.write(requirements)
        
        print(f"✅ Created {req_path}")
    
    def create_github_action(self):
        """Create GitHub Action for automatic deployment."""
        action_content = """name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Build site
        run: mkdocs build
      
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v2
        with:
          path: ./site
  
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v2
"""
        
        action_dir = self.output_dir / ".github" / "workflows"
        action_dir.mkdir(parents=True, exist_ok=True)
        action_path = action_dir / "deploy.yml"
        
        with open(action_path, 'w') as f:
            f.write(action_content)
        
        print(f"✅ Created GitHub Action")
    
    def create_api_pages(self):
        """Create mkdocstrings-backed API pages for key modules."""
        # Temporarily disabled due to path issues
        # api_dir = self.docs_dir / 'api'
        # api_dir.mkdir(parents=True, exist_ok=True)

        # Day 1 tile fetcher
        # with open(api_dir / 'day01_tile_fetcher.md', 'w') as f:
        #     f.write('# Day 1 Tile Fetcher API\n\n')
        #     f.write('::: src.day01_concurrency.tile_fetcher\n')

        # Day 3 FastAPI app
        # with open(api_dir / 'day03_api.md', 'w') as f:
        #     f.write('# Day 3 API\n\n')
        #     f.write('::: src.day03_api.app\n')

        print("✅ API pages temporarily disabled")

    def run(self):
        """Execute the conversion."""
        print("🚀 Starting MkDocs conversion...")
        
        self.setup_mkdocs_structure()
        self.create_index_page()
        # Pre-generate day pages so nav can reference existing files
        for day_dir in sorted(self.src_dir.glob("day*")):
            if day_dir.is_dir():
                self.process_day_directory(day_dir)
        self.create_api_pages()
        self.create_mkdocs_config()
        self.create_extra_files()
        self.create_requirements_txt()
        self.create_github_action()
        
        print("\n✨ Conversion complete!")
        print(f"\n📂 Output directory: {self.output_dir}")
        print("\n🎯 Next steps:")
        print("  1. cd " + str(self.output_dir))
        print("  2. pip install -r requirements.txt")
        print("  3. mkdocs serve  # View at http://localhost:8000")
        print("  4. mkdocs build  # Build static site")
        print("  5. mkdocs gh-deploy  # Deploy to GitHub Pages")


if __name__ == "__main__":
    converter = MkDocsConverter()
    converter.run()

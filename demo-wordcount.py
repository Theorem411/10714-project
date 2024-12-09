import json
import re

def count_words_in_notebook(file_path):
    # Load the .ipynb file
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    word_count = 0
    
    # Iterate over all cells
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') in ['markdown', 'raw']:  # Exclude code cells
            source = ''.join(cell.get('source', []))  # Get the text content
            words = re.findall(r'\b\w+\b', source)  # Tokenize into words
            word_count += len(words)  # Count words
    
    return word_count

# Example usage
file_path = "demo.ipynb"  # Replace with your file path
print(f"Word count (excluding code): {count_words_in_notebook(file_path)}")
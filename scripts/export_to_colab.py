import os
import zipfile
import json
import shutil

def update_notebooks():
    nb_dir = 'cervisense-ai/notebooks'
    if not os.path.exists(nb_dir):
        return
    for fname in os.listdir(nb_dir):
        if not fname.endswith('.ipynb'):
            continue
        path = os.path.join(nb_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'code':
                    new_source = []
                    for line in cell.get('source', []):
                        if '!git clone ' in line:
                            continue
                        if '%cd /content/cervisense-ai' in line:
                            new_source.append('!unzip -o -q /content/drive/MyDrive/cervisense-ai_export.zip -d /content/ > /dev/null\n')
                            new_source.append('!rm /content/drive/MyDrive/cervisense-ai_export.zip\n') # Clean up for Gemini
                            new_source.append('%cd /content/cervisense-ai\n')
                        else:
                            if '!unzip -o -q /content/drive/MyDrive/cervisense-ai_export.zip' in line or '!rm /content/drive/MyDrive/cervisense-ai_export.zip' in line:
                                    pass
                            else:
                                new_source.append(line)
                                
                    final_source = []
                    added_unzip = False
                    for line in new_source:
                        if '!unzip' in line:
                            if added_unzip: continue
                            added_unzip = True
                        final_source.append(line)
                    cell['source'] = final_source
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            print(f"Updated {fname} for offline drive extract setup.")
        except Exception as e:
            print(f"Error updating {fname}: {e}")

def export_project():
    project_dir = 'cervisense-ai'
    zip_path = 'cervisense-ai_export.zip'
    
    # 1. DELETE OLD 33GB ZIP
    if os.path.exists(zip_path):
        print(f"Deleting huge previous export zip: {zip_path}")
        os.remove(zip_path)
    if os.path.exists(os.path.join(project_dir, zip_path)):
        os.remove(os.path.join(project_dir, zip_path))
        
    update_notebooks()
    
    print(f"Creating highly optimized {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_dir):
            
            # Remove heavy, unneeded folders from the os.walk tree directly
            for skip_dir in ['data', 'cervisense_env', '.git', '__pycache__', 'venv', '.vscode', 'checkpoints', 'outputs']:
                if skip_dir in dirs:
                    dirs.remove(skip_dir)
                    
            for file in files:
                if file.endswith('.pyc') or file.endswith('.zip') or file.endswith('.pth'):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(project_dir))
                zipf.write(file_path, arcname)
                
    print(f"Export strictly clean. Size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")

if __name__ == '__main__':
    export_project()

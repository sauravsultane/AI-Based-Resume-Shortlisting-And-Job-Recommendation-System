import pandas as pd
import glob
import os

def consolidate():
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = glob.glob(os.path.join(dataset_dir, "naukri_*.csv"))
    
    # Exclude the combined file if it already exists to avoid recursion
    if os.path.join(dataset_dir, "naukri_combined_jobs.csv") in csv_files:
        csv_files.remove(os.path.join(dataset_dir, "naukri_combined_jobs.csv"))
        
    dfs = []
    
    print(f"Found {len(csv_files)} files to consolidate.")
    
    category_map = {
        'java': 'Java Developer',
        'php': 'PHP Developer',
        'python': 'Python Developer',
        'react': 'React Developer',
        'devops': 'DevOps Engineer',
        'datascience': 'Data Scientist',
        'mechanical': 'Mechanical Engineer',
        'civil': 'Civil Engineer',
        'electrical': 'Electrical Engineer',
    }

    for path in csv_files:
        filename = os.path.basename(path)
        print(f"Processing {filename}...")
        
        try:
            df = pd.read_csv(path)
            
            # Derive Category if not present
            if 'Category' not in df.columns:
                raw_cat = filename.replace('naukri_', '').replace('_jobs.csv', '')
                category = category_map.get(raw_cat, raw_cat.title() + ' Developer')
                df['Category'] = category
            
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(dataset_dir, "naukri_combined_jobs.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"Successfully created {output_path}")
        print(f"Total Records: {len(combined_df)}")
    else:
        print("No data found.")

if __name__ == "__main__":
    consolidate()

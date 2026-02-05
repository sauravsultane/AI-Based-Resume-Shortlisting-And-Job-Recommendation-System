import pandas as pd
import os

def filter_engineering_only():
    """
    Filter Resume.csv to keep only engineering and technical categories.
    Excludes: HR, Accountant, Advocate, Teacher, Sales, Healthcare, Arts, etc.
    """
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(dataset_dir, "Resume.csv")
    output_path = os.path.join(dataset_dir, "Resume_Engineering_Only.csv")
    
    # Load full dataset
    df = pd.read_csv(input_path)
    print(f"Total records in original dataset: {len(df)}")
    print(f"Total categories: {df['Category'].nunique()}")
    
    # Define engineering/technical categories to KEEP (based on actual Resume.csv categories)
    engineering_categories = [
        'ENGINEERING',
        'INFORMATION-TECHNOLOGY',
        'AUTOMOBILE',
        'AVIATION',
        'CONSTRUCTION',
        'DIGITAL-MEDIA',
        'DESIGNER'  # Technical design roles (UI/UX, Graphic Design)
        # EXCLUDED: ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, BANKING, 
        # BPO, BUSINESS-DEVELOPMENT, CHEF, CONSULTANT, FITNESS, FINANCE, 
        # HEALTHCARE, HR, PUBLIC-RELATIONS, SALES, TEACHER
    ]
    
    # Filter dataset (case-insensitive matching)
    engineering_categories_lower = [cat.lower() for cat in engineering_categories]
    df_filtered = df[df['Category'].str.lower().isin(engineering_categories_lower)]
    
    # Save filtered dataset
    df_filtered.to_csv(output_path, index=False)
    
    print(f"\n✓ Filtered dataset created: {output_path}")
    print(f"Records after filtering: {len(df_filtered)}")
    print(f"Categories after filtering: {df_filtered['Category'].nunique()}")
    print(f"\nCategories included:")
    for cat in sorted(df_filtered['Category'].unique()):
        count = len(df_filtered[df_filtered['Category'] == cat])
        print(f"  - {cat}: {count} resumes")

if __name__ == "__main__":
    filter_engineering_only()

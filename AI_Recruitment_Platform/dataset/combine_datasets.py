import pandas as pd
import os

def combine_resume_and_job_data():
    """
    Combine Resume_Engineering_Only.csv with naukri_combined_jobs.csv
    to create a richer training dataset.
    """
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load resume data
    resume_path = os.path.join(dataset_dir, "Resume_Engineering_Only.csv")
    df_resumes = pd.read_csv(resume_path)
    print(f"Loaded {len(df_resumes)} engineering resumes")
    
    # Load job data
    job_path = os.path.join(dataset_dir, "naukri_combined_jobs.csv")
    df_jobs = pd.read_csv(job_path)
    print(f"Loaded {len(df_jobs)} job descriptions")
    
    # Map job categories to resume categories
    category_mapping = {
        'Java Developer': 'INFORMATION-TECHNOLOGY',
        'PHP Developer': 'INFORMATION-TECHNOLOGY',
        'Python Developer': 'INFORMATION-TECHNOLOGY',
        'React Developer': 'INFORMATION-TECHNOLOGY',
        'DevOps Engineer': 'INFORMATION-TECHNOLOGY',
        'Data Scientist': 'INFORMATION-TECHNOLOGY',
        'Mechanical Engineer': 'ENGINEERING',
        'Civil Engineer': 'ENGINEERING',
        'Electrical Engineer': 'ENGINEERING'
    }
    
    # Create synthetic resume records from job descriptions
    synthetic_resumes = []
    for _, job in df_jobs.iterrows():
        # Map job category to resume category
        resume_category = category_mapping.get(job['Category'], None)
        
        if resume_category:
            # Create a synthetic resume text from job description
            synthetic_text = f"""
Job Title: {job['Title']}
Company: {job['Company']}
Location: {job['Location']}
Experience Required: {job['Experience']}

Job Description:
{job['Description']}

Required Skills:
{job.get('Skills', '')}
            """.strip()
            
            synthetic_resumes.append({
                'ID': f"JOB_{job.get('Title', 'Unknown')[:20]}",
                'Resume_str': synthetic_text,
                'Resume_html': '',
                'Category': resume_category
            })
    
    df_synthetic = pd.DataFrame(synthetic_resumes)
    print(f"Created {len(df_synthetic)} synthetic resume records from job descriptions")
    
    # Combine original resumes + synthetic resumes
    df_combined = pd.concat([df_resumes, df_synthetic], ignore_index=True)
    
    # Save combined dataset
    output_path = os.path.join(dataset_dir, "Resume_Engineering_Augmented.csv")
    df_combined.to_csv(output_path, index=False)
    
    print(f"\n✓ Combined dataset saved: {output_path}")
    print(f"Total records: {len(df_combined)}")
    print(f"  - Original resumes: {len(df_resumes)}")
    print(f"  - Synthetic (from jobs): {len(df_synthetic)}")
    print(f"\nCategory distribution:")
    print(df_combined['Category'].value_counts())

if __name__ == "__main__":
    combine_resume_and_job_data()

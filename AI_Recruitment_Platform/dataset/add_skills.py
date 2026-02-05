import pandas as pd
import os

def add_skills_column():
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(dataset_dir, "naukri_combined_jobs.csv")
    
    # Skill mapping based on Category
    skill_map = {
        'Java Developer': 'Java, Spring Boot, Hibernate, Maven, SQL, REST API, Microservices',
        'PHP Developer': 'PHP, Laravel, MySQL, JavaScript, HTML, CSS, Apache, REST API',
        'Python Developer': 'Python, Django, Flask, REST API, PostgreSQL, Git, Docker, AWS',
        'React Developer': 'React, JavaScript, TypeScript, HTML, CSS, Redux, Node.js, Webpack',
        'DevOps Engineer': 'AWS, Docker, Kubernetes, Jenkins, CI/CD, Linux, Terraform, Ansible',
        'Data Scientist': 'Python, Machine Learning, Pandas, NumPy, Scikit-learn, TensorFlow, SQL, Statistics',
        'Mechanical Engineer': 'AutoCAD, SolidWorks, ANSYS, CATIA, Manufacturing, Thermodynamics, CAD/CAM',
        'Civil Engineer': 'AutoCAD, Revit, STAAD Pro, Project Management, Structural Analysis, Surveying',
        'Electrical Engineer': 'Circuit Design, PLC, SCADA, AutoCAD Electrical, Power Systems, Embedded Systems'
    }
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Add Skills column if it doesn't exist
    if 'Skills' not in df.columns:
        print("Adding 'Skills' column...")
        
        # Map skills based on Category
        df['Skills'] = df['Category'].map(skill_map).fillna('Communication, Teamwork, Problem Solving')
        
        # Save back to CSV
        df.to_csv(csv_path, index=False)
        print(f"Successfully added 'Skills' column to {csv_path}")
        print(f"Total records: {len(df)}")
    else:
        print("'Skills' column already exists. Updating...")
        df['Skills'] = df['Category'].map(skill_map).fillna('Communication, Teamwork, Problem Solving')
        df.to_csv(csv_path, index=False)
        print(f"Successfully updated 'Skills' column")

if __name__ == "__main__":
    add_skills_column()

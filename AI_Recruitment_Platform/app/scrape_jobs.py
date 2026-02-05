try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
except ImportError:
    webdriver = None
from dashboard.models import Job
import time
import random

def run_scraper(pages=1):
    """
    Scrape jobs from Naukri using Selenium.
    Note: Naukri has heavy anti-bot protection. This is a basic implementation.
    """
    print("Starting Job Scraper...")
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run invisible
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Add fake user agent to bypass basic checks
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    driver = webdriver.Chrome(options=chrome_options)
    
    jobs_scraped = 0
    
    try:
        for i in range(1, pages + 1):
            url = f"https://www.naukri.com/python-developer-jobs-{i}"
            print(f"Scraping {url}...")
            driver.get(url)
            
            # Wait for content
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "cust-job-tuple"))
                )
            except:
                print(f"Timeout waiting for page {i}")
                continue

            job_elements = driver.find_elements(By.CLASS_NAME, "cust-job-tuple")
            
            for job_elem in job_elements:
                try:
                    title = job_elem.find_element(By.CLASS_NAME, "title").text
                    company = job_elem.find_element(By.CLASS_NAME, "comp-name").text
                    
                    try:
                        location = job_elem.find_element(By.CLASS_NAME, "loc-wrap").text
                    except:
                        location = "Remote/Unknown"
                        
                    try:
                        salary = job_elem.find_element(By.CLASS_NAME, "sal-wrap").text
                    except:
                        salary = "Not Disclosed"
                        
                    try:
                        experience = job_elem.find_element(By.CLASS_NAME, "exp-wrap").text
                    except:
                        experience = "0-1 Yrs"
                        
                    try:
                        description = job_elem.find_element(By.CLASS_NAME, "job-desc").text
                    except:
                        description = title
                        
                    # Save to DB if not exists
                    if not Job.objects.filter(title=title, company=company).exists():
                        Job.objects.create(
                            title=title,
                            company=company,
                            location=location,
                            salary=salary,
                            experience=experience,
                            description=description,
                            required_skills="Python, Django" # Placeholder
                        )
                        jobs_scraped += 1
                        
                except Exception as e:
                    print(f"Error parsing job info: {e}")
                    continue
                    
            time.sleep(random.uniform(2, 5)) # Random delay
            
    except Exception as e:
        print(f"Scraper Error: {e}")
    finally:
        driver.quit()
        
    print(f"Scraping Completed. Added {jobs_scraped} new jobs.")
    return jobs_scraped

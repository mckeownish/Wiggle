
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import glob

def send_WAXSiS(full_pdb_path):

    # Initialize the WebDriver
    driver = webdriver.Chrome()  # Or use webdriver.Firefox()

    try:
        # Open the WAXSiS webpage
        driver.get("http://waxsis.uni-saarland.de/")

        # Wait for and click the "Review Options" button
        review_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "review_link"))
        )
        review_button.click()

        # Wait for the upload button and upload the PDB file
        upload_button = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "upload_pdb"))
        )
        upload_button.send_keys(full_pdb_path)

        # Wait for the "Remove all ligands" checkbox to appear and click it
        remove_ligands_checkbox = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.ID, "ligand3"))
        )
        remove_ligands_checkbox.click()

        # Submit the job
        submit_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.NAME, "submit"))
        )
        submit_button.click()

        # Wait for the result link to appear
        result_link = WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.XPATH, "//a[contains(@href, 'jobs/results')]"))
        )
        result_url = result_link.get_attribute("href")

        # Print the result URL
        print("Result Link:", result_url)

    except Exception as e:
        print("An error occurred:", e)

    finally:
        # Close the browser
        driver.quit()

    return result_url


def download_WAXSiS_result(result_url, download_directory):
    options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": download_directory,
        "download.prompt_for_download": False,
        "directory_upgrade": True,
    }
    options.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome(options=options)
    try:
        # Get initial file count
        initial_files = set(glob.glob(os.path.join(download_directory, "*")))
        
        driver.get(result_url)
        download_link = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.LINK_TEXT, "Click here to download the complete results collection"))
        )
        download_link.click()
        
        # Wait for new file to appear and download to complete
        def download_completed():
            current_files = set(glob.glob(os.path.join(download_directory, "*")))
            new_files = current_files - initial_files
            if not new_files:
                return False
            return not any(f.endswith('.crdownload') or f.endswith('.tmp') for f in new_files)
        
        WebDriverWait(driver, 300).until(lambda x: download_completed())
        print("Download completed successfully.")
        
    except Exception as e:
        print("An error occurred:", e)
    finally:
        driver.quit()
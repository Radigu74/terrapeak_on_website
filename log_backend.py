import csv
import os

def save_user_data(name, email, phone, country):
    """
    Saves the user details into a CSV file called 'user_logs.csv' 
    on the persistent volume mounted at /data.
    """
    # Use the mount path provided by Railway
    file_path = '/data/user_logs.csv'
    
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'email', 'phone', 'country']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'name': name,
            'email': email,
            'phone': phone,
            'country': country
        })

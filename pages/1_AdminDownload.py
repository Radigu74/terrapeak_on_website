import streamlit as st
import os

def main():
    st.title("Admin Download")
    st.write("Enter the admin password to download the user logs CSV.")

    admin_pass = st.text_input("Admin Password", type="password")
    if admin_pass == "Terrapeak2025":  # Replace with your actual secret password
        file_path = "/data/user_logs.csv"  # Adjust this path if needed
        try:
            with open(file_path, "rb") as f:
                file_data = f.read()
            st.download_button(
                label="Download User Logs CSV",
                data=file_data,
                file_name="user_logs.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error reading the file: {e}")
    else:
        st.info("Please enter the admin password.")

if __name__ == "__main__":
    main()

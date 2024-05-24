import os
from openpyxl import Workbook

def extract_folder_names(directory):
    folder_names = []
    for entry in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, entry)):
            folder_names.append(entry)
    return folder_names

def save_to_excel(folder_names, excel_file):
    wb = Workbook()
    ws = wb.active
    ws.title = "Folder Names"
    for i, folder_name in enumerate(folder_names, start=2):  # Start from the second row
        ws.cell(row=i, column=1, value=folder_name)
    wb.save(excel_file)
    print(f"Folder names saved to {excel_file}")

if __name__ == "__main__":
    directory = "MTCNN_Face_Dataset"  # for MTCNN
    # directory = "Haar_Face_Dataset" #for Haar-cascascade
    excel_file = "attendancesheet.xlsx"  # Change this to the desired Excel file name

    folder_names = extract_folder_names(directory)
    print("Folder names extracted:", folder_names)  # Print extracted folder names
    save_to_excel(folder_names, excel_file)

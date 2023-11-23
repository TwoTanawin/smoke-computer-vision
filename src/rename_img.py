import os

folder_path = '/app/smoke-detect/video/data-23-11-23'
file_prefix = 'vdo'

file_list = os.listdir(folder_path)
file_list.sort()  # Sort the file list in alphabetical order

for index, filename in enumerate(file_list):
    file_ext = os.path.splitext(filename)[1]  # Get the file extension
    new_filename = f'{file_prefix}{index+1:03d}{file_ext}'  # Generate the new filename with leading zeros
    destination_path = os.path.join(folder_path, new_filename)

    # Check if the destination file already exists
    if os.path.isfile(destination_path):
        suffix = 1
        while True:
            # Generate a unique filename by appending a suffix
            unique_filename = f'{file_prefix}{index+1:03d}_{suffix}{file_ext}'
            destination_path = os.path.join(folder_path, unique_filename)

            if not os.path.isfile(destination_path):
                break
            suffix += 1

    os.rename(os.path.join(folder_path, filename), destination_path)

print("Done!")

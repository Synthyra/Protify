"""Delete the files in the run folders (logs, results, plots, embeddings, weights) after asking per folder.

From a shell: `python -m protify.clean_run_folders`.
"""

import os


def main() -> None:
    folders_to_clean = ['logs', 'results', 'plots', 'embeddings', 'weights']

    for folder in folders_to_clean:
        if os.path.exists(folder):
            files = os.listdir(folder)
            if files:
                response = input(f"Do you want to delete all files in '{folder}' folder? ({len(files)} files) [y/N]: ")
                if response.lower() == 'y':
                    for file in files:
                        file_path = os.path.join(folder, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    print(f"All files in '{folder}' have been deleted.")
                else:
                    print(f"Skipped cleaning '{folder}' folder.")
            else:
                print(f"'{folder}' folder is already empty.")
        else:
            print(f"'{folder}' folder does not exist.")


if __name__ == "__main__":
    main()

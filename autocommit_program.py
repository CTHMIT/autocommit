import subprocess
import os
from typing import List, Optional
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
import sys

load_dotenv()


def generate_commit_message(llm: OllamaLLM, content: str) -> Optional[str]:
    """
    Generate a commit message using the provided LLM model based on the given diff content.
    """
    system_prompt = f"""
    Only use the following information to answer the question.
    - Do not use external knowledge or assumptions.
    - Focus strictly on the changes in the diff.
    - Be concise, specific, and accurate.
    Task: Write a Git commit message that clearly describes the changes in the following diff:
    ```
    {content}
    ```
    """
    try:
        return llm.invoke(system_prompt).strip()
    except Exception as e:
        print(f"Error generating commit message: {e}")
        return None


def get_staged_files() -> List[str]:
    """
    Retrieve a list of currently staged files.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--cached"], capture_output=True, text=True, check=True
        )
        return result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Error getting staged files: {e}")
        return []


def get_file_diff(repo_path: str, file_path: str) -> Optional[str]:
    """
    Retrieve the diff of a specific file from the staged changes.
    """
    try:
        diff_content = subprocess.check_output(
            ["git", "-C", repo_path, "diff", "--cached", file_path], encoding="utf-8"
        )
        return diff_content if diff_content else None
    except subprocess.CalledProcessError as e:
        print(f"Error getting diff for {file_path}: {e}")
        return None


def main(commit_message_file: str):
    repo_path = os.getcwd()
    model_name = "llama3.2"
    llm = OllamaLLM(model=model_name)

    # Get all staged files.
    staged_files = get_staged_files()
    if not staged_files:
        print("No staged files found.")
        return

    # Generate commit messages for all staged files.
    combined_diff = ""
    for file_path in staged_files:
        file_diff = get_file_diff(repo_path, file_path)
        if file_diff:
            combined_diff += file_diff

    if not combined_diff:
        print("No diffs found for any staged files.")
        return

    # Generate the commit message based on the combined diffs.
    commit_message = generate_commit_message(llm, combined_diff)
    if not commit_message:
        print("Failed to generate commit message.")
        return

    # Write the generated commit message to the commit message file.
    try:
        with open(commit_message_file, 'w') as f:
            f.write(commit_message)
        print(f"Commit message generated and written to {commit_message_file}:\n{commit_message}")
    except Exception as e:
        print(f"Error writing commit message to {commit_message_file}: {e}")


if __name__ == "__main__":
    # Pre-commit passes the commit message file path as the first argument.
    if len(sys.argv) > 1:
        commit_message_file = sys.argv[1]
        main(commit_message_file)
    else:
        print("No commit message file provided. Exiting.")

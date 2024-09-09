import os
import requests
from pinecone import Pinecone, ServerlessSpec
from transformers import RobertaTokenizer, RobertaModel
import torch
import base64

# Initialize Pinecone
def initialize_pinecone(index_name, dimension=768):
    # Debug print statement
    print("Inside initialize_pinecone")

    pc = Pinecone(api_key=os.environ.get("PINECONE_KEY"))

    # Now do stuff
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    index = pc.Index(index_name)
    return index

# Function to extract GitHub username from the profile URL
def extract_username(github_url):
    # Debug print statement
    print("Inside extract username")

    if "github.com/" in github_url:
        return github_url.split("github.com/")[1].split('/')[0]
    else:
        raise ValueError("Invalid GitHub profile URL")

# Function to get repositories for a user with GitHub token for authentication
def get_repositories(username, token):
    # Debug print statement
    print("Inside get_repositories")

    url = f"https://api.github.com/users/{username}/repos"
    headers = {'Authorization': f'token {token}'}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch repositories for {username}, status code: {response.status_code}")
    
    repos = response.json()
    return repos

# Function to prioritize files based on their extension, with GitHub token for authentication
def prioritize_files(username, repo_name, token):
    # Debug print statement
    print("Inside prioritize_files")

    branches = ['main', 'master']
    files = []
    
    for branch in branches:
        tree_url = f"https://api.github.com/repos/{username}/{repo_name}/git/trees/{branch}?recursive=1"
        headers = {'Authorization': f'token {token}'}
        
        response = requests.get(tree_url, headers=headers)
        
        if response.status_code == 200:
            files = response.json().get('tree', [])
            print(f"Successfully fetched file tree from branch: {branch}")
            break
        else:
            print(f"Failed to fetch file tree from branch: {branch}, status code: {response.status_code}")
            if response.status_code == 404 and branch == 'main':
                # If 'main' branch fails, continue to check 'master'
                continue
            elif response.status_code != 404:
                # If the error is not a 404, raise an exception
                raise Exception(f"Failed to fetch file tree for {repo_name} from branch {branch}, status code: {response.status_code}")

    if not files:
        raise Exception(f"Failed to fetch file tree for {repo_name} from both branches: 'main' and 'master'")
    
    code_files = []
    
    # Define file extensions to prioritize
    prioritized_extensions = ['.py', '.js', '.java', '.cpp', '.c', '.ts', '.rb', '.php']

    # Process the files and filter/prioritize
    for file in files:
        if file['type'] == 'blob':  # Ensure it's a file, not a directory
            file_path = file['path']
            file_size_kb = file.get('size', 0) / 1024
            
            # Prioritize based on extension
            if any(file_path.endswith(ext) for ext in prioritized_extensions):
                # Skip overly large files (example: more than 1 MB)
                if file_size_kb < 1 * 1024:
                    code_files.append({'file': file_path, 'size_kb': file_size_kb})

    return code_files


# Function to fetch the content of a file from GitHub
def get_file_content(username, repo_name, file_path, token):
    # Debug print statement
    print("Inside get_file_content")

    url = f"https://api.github.com/repos/{username}/{repo_name}/contents/{file_path}"
    headers = {'Authorization': f'token {token}'}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json().get('content', '')
        # Decode base64 content
        content = base64.b64decode(content).decode('utf-8')
        return content
    else:
        print(f"Failed to fetch content for {file_path}, status code: {response.status_code}")
        return None

# Function to initialize CodeBERT model
def initialize_codebert():
    # Debug print statement
    print("Inside initialize_codebert")

    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = RobertaModel.from_pretrained('microsoft/codebert-base')
    return tokenizer, model

# Function to convert text to vector embeddings using CodeBERT
def get_embedding(text, tokenizer, model):
    # Debug print statement
    print("Inside get_embedding")

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean of the hidden states as the embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Function to store prioritized files in Pinecone
def store_prioritized_files_in_pinecone(username, repo_name, prioritized_files, token, pinecone_index, tokenizer, model):
    # Debug print statement
    print("Inside store_prioritized_files_in_pinecone")

    for file_info in prioritized_files:
        file_path = file_info['file']
        file_content = get_file_content(username, repo_name, file_path, token)
        
        if file_content:
            # Convert file content to vector embedding using CodeBERT
            embedding = get_embedding(file_content, tokenizer, model)
            # Create a unique ID for the file (can be a combination of repo name + file path)
            vector_id = f"{repo_name}/{file_path}"
            
            # Store the embedding in Pinecone along with metadata
            pinecone_index.upsert(vectors=[(vector_id, embedding)], metadata={'repo_name': repo_name, 'file_path': file_path})

# Main function to handle GitHub profile analysis and Pinecone storage
def analyze_and_store_in_pinecone(profile_url, token, pinecone_index_name):
    # Debug print statement
    print("Inside analyze_and_store_in_pinecone")

    try:
        pinecone_index = initialize_pinecone(pinecone_index_name)
        tokenizer, model = initialize_codebert()
        
        username = extract_username(profile_url)
        repos = get_repositories(username, token)
        
        for repo in repos:
            repo_name = repo['name']
            print(f"Processing repository: {repo_name}")
            
            prioritized_files = prioritize_files(username, repo_name, token)
            
            # Debug print statement
            print(f"Number of prioritized files: {len(prioritized_files)}")
            
            store_prioritized_files_in_pinecone(username, repo_name, prioritized_files, token, pinecone_index, tokenizer, model)
            print(f"Stored {len(prioritized_files)} files from {repo_name} in Pinecone.")
    
    except Exception as e:
        print(f"Error: {e}")


# Example usage
if __name__ == "__main__":
    github_profile_url = input("Enter GitHub profile URL: ")
    # github_profile_url = "https://github.com/Akki-58/"
    github_token = os.getenv("GIT_KEY") # Input GitHub token
    pinecone_index_name = "github-code"  # Define a Pinecone index name
    
    analyze_and_store_in_pinecone(github_profile_url, github_token, pinecone_index_name)
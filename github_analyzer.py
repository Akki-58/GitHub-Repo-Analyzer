from dotenv import load_dotenv
import os
# import langchain
# print(langchain.__version__)

import requests
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# GitHub API endpoint
API_ENDPOINT = "https://api.github.com"
FILE_SIZE_THRESHOLD = 1024 * 1024 * 5  # 5 MB file size limit

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
index = pc.Index("github-code")  # Ensure this matches your Pinecone setup

# Load CodeBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# Cache to store repository details
repo_cache = {}

def get_user_repos(username, token=None):
    """
    Fetch the list of public repositories for a given GitHub username.
    """
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    
    url = f"{API_ENDPOINT}/users/{username}/repos?per_page=100"
    response = requests.get(url, headers=headers)
    repos = response.json()

    while 'next' in response.links:
        next_url = response.links['next']['url']
        response = requests.get(next_url, headers=headers)
        repos.extend(response.json())

    return repos

def get_repo_details(repo_url, token=None):
    """
    Fetch the detailed information of a repository.
    """
    if repo_url in repo_cache:
        return repo_cache[repo_url]
    
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    
    response = requests.get(repo_url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching repo details: {response.status_code}")
        return {}
    
    repo_details = response.json()
    repo_cache[repo_url] = repo_details
    return repo_details

def get_repo_languages(repo_url, token=None):
    """
    Fetch the programming languages used in a repository.
    """
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    
    url = f"{repo_url}/languages"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching languages: {response.status_code}")
        return {}
    
    languages = response.json()
    return languages

def get_repo_contributors(repo_url, token=None):
    """
    Fetch the contributors of a repository.
    """
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    
    url = f"{repo_url}/contributors"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching contributors: {response.status_code}")
        return []
    
    contributors = response.json()
    return contributors

def get_repo_commits(repo_url, token=None):
    """
    Fetch the commit history of a repository.
    """
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    
    url = f"{repo_url}/commits"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching commits: {response.status_code}")
        return []
    
    commits = response.json()
    return commits

def fetch_repo_files(repo_owner, repo_name, path='', token=None):
    """
    Recursively fetch all code files (.py, .js, .cpp) from a GitHub repository, including files inside folders.
    """
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'

    url = f"{API_ENDPOINT}/repos/{repo_owner}/{repo_name}/contents/{path}"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: Unable to fetch contents for {path}. Status Code: {response.status_code}")
        return []
    
    contents = response.json()
    code_files = []

    for item in contents:
        if item['type'] == 'file' and item['name'].endswith(('.py', '.js', '.cpp', '.c', '.java')):
            # Check file size
            if item['size'] <= FILE_SIZE_THRESHOLD:
                code_files.append(item['download_url'])
        elif item['type'] == 'dir':
            # Recursively fetch files from subdirectories
            code_files.extend(fetch_repo_files(repo_owner, repo_name, path=item['path'], token=token))

    return code_files

def embed_code(code):
    """
    Generate an embedding for a code file using CodeBERT.
    """
    inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Pooling to get a single vector
    return embeddings.detach().numpy()[0]

def push_to_pinecone(file_urls, repo_name):
    """
    Fetch file contents, embed them, and push to Pinecone.
    """
    for file_url in file_urls:
        response = requests.get(file_url)
        if response.status_code == 200:
            code_content = response.text
            embedding = embed_code(code_content)
            # Push to Pinecone
            index.upsert(vectors=[(file_url, embedding)], namespace=repo_name)

def scrape_github_profile(username, token=None):
    """
    Scrape the public repositories of a GitHub user and retrieve the necessary details.
    """
    repos = get_user_repos(username, token=token)
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for repo in repos:
            repo_url = repo["url"]
            futures.append(executor.submit(get_repo_details, repo_url, token))
        
        for future in futures:
            repo_details = future.result()
            if not repo_details:
                continue
            
            repo_name = repo_details.get("name", "Unknown")
            repo_description = repo_details.get("description", "No description")
            repo_stars = repo_details.get("stargazers_count", 0)
            repo_forks = repo_details.get("forks_count", 0)
            repo_watchers = repo_details.get("subscribers_count", 0)
            repo_size = repo_details.get("size", 0)
            repo_owner = repo_details["owner"]["login"]
            
            languages = get_repo_languages(repo_details["url"], token=token)
            contributors = get_repo_contributors(repo_details["url"], token=token)
            commits = get_repo_commits(repo_details["url"], token=token)
            code_files = fetch_repo_files(repo_owner, repo_name, token=token)

            # Push the code files to Pinecone
            push_to_pinecone(code_files, repo_name)

            # Print the repository details
            print(f"Repository: {repo_name}")
            print(f"Description: {repo_description}")
            print(f"Stars: {repo_stars}, Forks: {repo_forks}, Watchers: {repo_watchers}")
            print(f"Primary Language: {list(languages.keys())[0] if languages else 'Unknown'}")
            print(f"Contributors: {len(contributors)}, Commits: {len(commits)}")
            print(f"Size: {repo_size} KB")
            print(f"Code Files: {len(code_files)}")
            print("---")

# Example usage
# username1 = "Akki-58"
# username2 = "kanakmaheshwari3115"
# token = os.getenv("GIT_KEY")
# scrape_github_profile(username1, token)  # Replace with a real GitHub username and token

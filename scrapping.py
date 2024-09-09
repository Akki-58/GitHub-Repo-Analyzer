import requests
import os

# Function to extract GitHub username from the profile URL
def extract_username(github_url):
    if "github.com/" in github_url:
        return github_url.split("github.com/")[1].split('/')[0]
    else:
        raise ValueError("Invalid GitHub profile URL")

# Function to get repositories for a user with GitHub token for authentication
def get_repositories(username, token):
    url = f"https://api.github.com/users/{username}/repos"
    headers = {'Authorization': f'token {token}'}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch repositories for {username}, status code: {response.status_code}")
    
    repos = response.json()
    repo_details = []
    
    for repo in repos:
        details = {
            'name': repo['name'],
            'description': repo['description'],
            'stars': repo['stargazers_count'],
            'forks': repo['forks_count'],
            'watchers': repo['watchers_count'],
            'primary_language': repo['language'],
            'repo_size_kb': repo['size'],
        }
        
        # Fetch additional languages used
        languages = requests.get(repo['languages_url'], headers=headers).json()
        details['languages'] = list(languages.keys())
        
        # Add contributors count
        contributors = len(requests.get(repo['contributors_url'], headers=headers).json())
        details['contributors'] = contributors
        
        # Add commits count
        commits_url = repo['commits_url'].replace("{/sha}", "")
        commits = len(requests.get(commits_url, headers=headers).json())
        details['commits'] = commits
        
        # Prioritize files in the repository
        prioritized_files = prioritize_files(username, repo['name'], token)
        details['prioritized_files'] = prioritized_files
        
        repo_details.append(details)
    
    return repo_details

# Function to prioritize files based on their extension, with GitHub token for authentication
def prioritize_files(username, repo_name, token):
    tree_url = f"https://api.github.com/repos/{username}/{repo_name}/git/trees/main?recursive=1"
    headers = {'Authorization': f'token {token}'}
    
    response = requests.get(tree_url, headers=headers)
    if response.status_code != 200:
        return []
    
    files = response.json().get('tree', [])
    code_files = []
    
    # Define file extensions to prioritize
    prioritized_extensions = ['.py', '.js', '.java', '.cpp', '.c', '.ts', '.rb', '.php']
    # deprioritized_extensions = ['.json', '.csv', '.md', '.ipynb', '.txt', '.yml']

    # Process the files and filter/prioritize
    for file in files:
        file_path = file['path']
        file_size_kb = file['size'] / 1024 if 'size' in file else 0

        # Prioritize based on extension
        if any(file_path.endswith(ext) for ext in prioritized_extensions):
            # Skip overly large files (example: more than 5 MB)
            if file_size_kb < 5*1024:
                code_files.append({'file': file_path, 'size_kb': file_size_kb})
        # elif not any(file_path.endswith(ext) for ext in deprioritized_extensions):
        #     # Include files that are neither deprioritized nor prioritized, but skip large ones
        #     if file_size_kb < 512:  # Limit for non-prioritized files
        #         code_files.append({'file': file_path, 'size_kb': file_size_kb})

    return code_files

# Main function to handle GitHub profile analysis with token
def analyze_github_profile(profile_url, token):
    try:
        username = extract_username(profile_url)
        repo_data = get_repositories(username, token)
        return repo_data
    except Exception as e:
        return str(e)

# Example usage
if __name__ == "__main__":
    # github_profile_url = input("Enter GitHub profile URL: ")
    github_profile_url = "https://github.com/Akki-58/"
    github_token = os.getenv("GIT_KEY")
    data = analyze_github_profile(github_profile_url, github_token)
    
    if isinstance(data, str):
        print(f"Error: {data}")
    else:
        # Print out the scraped data
        for repo in data:
            print(f"Repository: {repo['name']}")
            print(f"  Description: {repo['description']}")
            print(f"  Stars: {repo['stars']}, Forks: {repo['forks']}, Watchers: {repo['watchers']}")
            print(f"  Primary Language: {repo['primary_language']}")
            print(f"  Other Languages: {repo['languages']}")
            print(f"  Contributors: {repo['contributors']}, Commits: {repo['commits']}")
            print(f"  Repo Size: {repo['repo_size_kb']} KB")
            # print("  Prioritized Files:")
            # for file in repo['prioritized_files']:
            #     print(f"    {file['file']} (Size: {file['size_kb']} KB)")
            print()

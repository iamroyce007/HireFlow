import httpx
import re
import asyncio
from datetime import datetime, timedelta

class GitHubVerifier:
    """
    Advanced GitHub Verification Service.
    Checks for existence, activity levels, and identity alignment.
    """
    
    def __init__(self):
        self.headers = {"Accept": "application/vnd.github.v3+json"}

    async def verify_profile(self, github_url: str, candidate_name: str = "") -> dict:
        """
        Verify a GitHub profile and return a detailed report.
        """
        if not github_url or "github.com" not in github_url:
            return {"verified": False, "reason": "Invalid or missing GitHub URL"}

        # Extract username
        match = re.search(r"github\.com/([^/?#]+)", github_url)
        if not match:
            return {"verified": False, "reason": "Could not parse GitHub username"}
        
        username = match.group(1)
        
        results = {
            "username": username,
            "exists": False,
            "active": False,
            "identity_match": False,
            "reputation_score": 0,
            "verified": False,
            "checks": {}
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # 1. Existence & Basic Info
                user_res = await client.get(f"https://api.github.com/users/{username}", headers=self.headers)
                if user_res.status_code != 200:
                    results["reason"] = f"GitHub user not found (HTTP {user_res.status_code})"
                    return results
                
                user_data = user_res.json()
                results["exists"] = True
                results["checks"]["existence"] = "Passed"
                
                # 2. Identity Alignment
                full_name = user_data.get("name", "")
                bio = user_data.get("bio", "")
                
                if candidate_name and full_name:
                    # Simple fuzzy match (case-insensitive)
                    if candidate_name.lower() in full_name.lower() or full_name.lower() in candidate_name.lower():
                        results["identity_match"] = True
                        results["checks"]["identity"] = "Matches Candidate"
                    else:
                        results["checks"]["identity"] = "Mismatched (Profile name: " + full_name + ")"
                else:
                    results["checks"]["identity"] = "Unable to verify (Missing name on profile)"

                # 3. Activity Level & Complexity (Recent events + Repos)
                events_res = await client.get(f"https://api.github.com/users/{username}/events/public", headers=self.headers)
                repos_res = await client.get(f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated", headers=self.headers)
                
                if events_res.status_code == 200:
                    events = events_res.json()
                    if events:
                        last_event_date = datetime.strptime(events[0]["created_at"], "%Y-%m-%dT%H:%M:%SZ")
                        if datetime.utcnow() - last_event_date < timedelta(days=90):
                            results["active"] = True
                            results["checks"]["activity"] = "Active (Last contribution: " + events[0]["created_at"] + ")"
                        else:
                            results["checks"]["activity"] = "Inactive (Last activity > 90 days ago)"
                    else:
                        results["checks"]["activity"] = "No public activity found"
                
                if repos_res.status_code == 200:
                    repos = repos_res.json()
                    original_repos = [r for r in repos if not r.get("fork", False)]
                    
                    total_stars = sum(r.get("stargazers_count", 0) for r in original_repos)
                    total_size = sum(r.get("size", 0) for r in original_repos)
                    doc_count = sum(1 for r in original_repos if r.get("has_wiki") or r.get("has_issues"))
                    
                    complexity_score = min((total_size / 2000) + (doc_count * 2) + (total_stars * 0.5), 100)
                    
                    results["checks"]["complexity"] = {
                        "score": round(complexity_score, 1),
                        "original_repos": len(original_repos),
                        "total_stars": total_stars,
                        "codebase_size_kb": total_size,
                        "documented_projects": doc_count
                    }
                else:
                    results["checks"]["complexity"] = "Unable to fetch repository data"

                # 4. Reputation Score (0-100)
                reputation = 20 # Base for existing
                reputation += min(user_data.get("followers", 0) * 2, 30)
                reputation += min(user_data.get("public_repos", 0) * 2, 20)
                # Age bonus (if created > 1 year ago)
                created_at = datetime.strptime(user_data["created_at"], "%Y-%m-%dT%H:%M:%SZ")
                if datetime.utcnow() - created_at > timedelta(days=365):
                    reputation += 30
                
                results["reputation_score"] = min(reputation, 100)
                
                # Final decision
                # A profile is "verified" if it exists and either matches identity OR shows significant activity
                if results["exists"] and (results["active"] or results["identity_match"]):
                    results["verified"] = True
                    
                return results

        except Exception as e:
            results["reason"] = f"Extraction error: {str(e)}"
            return results

if __name__ == "__main__":
    # Test stub
    async def test():
        v = GitHubVerifier()
        # Test with a known active profile (e.g., github itself or common users)
        print("Verifying 'google' profile...")
        res = await v.verify_profile("https://github.com/google", "Google")
        import json
        print(json.dumps(res, indent=2))
        
    asyncio.run(test())

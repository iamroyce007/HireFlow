import re
import pdfplumber
import io
from typing import Dict, Optional

class LinkExtractor:
    """Service to extract professional links (GitHub, LinkedIn) from PDF resumes."""

    # Requested Regex Patterns
    GITHUB_REGEX = r"(https?://)?(www\.)?github\.com/[A-Za-z0-9_-]+"
    LINKEDIN_REGEX = r"(https?://)?(www\.)?linkedin\.com/in/[A-Za-z0-9_-]+"

    @staticmethod
    def extract_links_from_pdf(file_bytes: bytes) -> Dict[str, Optional[str]]:
        """Extracts text from PDF and returns the first valid GitHub and LinkedIn links."""
        try:
            text = ""
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            # Find all matches
            github_matches = re.findall(LinkExtractor.GITHUB_REGEX, text, re.IGNORECASE)
            linkedin_matches = re.findall(LinkExtractor.LINKEDIN_REGEX, text, re.IGNORECASE)

            # Normalization function
            def normalize_link(raw_match, full_text):
                # re.findall with groups returns tuples if there are multiple groups
                # We want the FULL match. Let's use re.finditer instead for accuracy.
                pass

            # Redoing with finditer for full string match
            github_link = None
            github_match = re.search(LinkExtractor.GITHUB_REGEX, text, re.IGNORECASE)
            if github_match:
                url = github_match.group(0)
                if not url.startswith('http'):
                    url = 'https://' + url
                github_link = url

            linkedin_link = None
            linkedin_match = re.search(LinkExtractor.LINKEDIN_REGEX, text, re.IGNORECASE)
            if linkedin_match:
                url = linkedin_match.group(0)
                if not url.startswith('http'):
                    url = 'https://' + url
                linkedin_link = url

            return {
                "github": github_link,
                "linkedin": linkedin_link
            }
        except Exception as e:
            print(f"[LinkExtractor] Error: {e}")
            return {"github": None, "linkedin": None}

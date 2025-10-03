#!/usr/bin/env python3
"""
Script to create a submission from a CoSci GitHub PR for paperbench evaluation.

This script:
1. Fetches PR information from GitHub
2. Extracts Claude's comment and associated job details
3. Downloads the repository at the PR branch state
4. Generates a submission structure compatible with run_judge.py
"""

import argparse
import asyncio
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


def run_command(cmd: list[str], cwd: Path | None = None) -> tuple[str, str, int]:
    """Run a shell command and return stdout, stderr, and return code."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.stdout, result.stderr, result.returncode


def fetch_pr_data(repo: str, pr_number: int) -> dict[str, Any]:
    """Fetch PR data from GitHub API."""
    logger.info("Fetching PR data", repo=repo, pr_number=pr_number)
    stdout, stderr, code = run_command(["gh", "api", f"repos/{repo}/pulls/{pr_number}"])
    if code != 0:
        raise RuntimeError(f"Failed to fetch PR data: {stderr}")
    return json.loads(stdout)


def fetch_pr_comments(repo: str, pr_number: int) -> list[dict[str, Any]]:
    """Fetch all comments on the PR."""
    logger.info("Fetching PR comments", repo=repo, pr_number=pr_number)
    stdout, stderr, code = run_command(
        ["gh", "api", f"repos/{repo}/issues/{pr_number}/comments"]
    )
    if code != 0:
        raise RuntimeError(f"Failed to fetch PR comments: {stderr}")
    return json.loads(stdout)


def find_claude_comment(comments: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find the Claude bot comment with job information."""
    for comment in comments:
        user = comment.get("user", {})
        if user.get("login") == "claude[bot]":
            body = comment.get("body", "")
            if "View job" in body and "actions/runs/" in body:
                return comment
    return None


def extract_job_url_from_comment(comment: dict[str, Any]) -> str | None:
    """Extract the GitHub Actions job URL from Claude's comment."""
    body = comment.get("body", "")
    # Look for pattern like [View job](https://github.com/.../actions/runs/...)
    import re

    match = re.search(r"\[View job\]\((https://github\.com/[^)]+/actions/runs/\d+)\)", body)
    if match:
        return match.group(1)
    return None


def fetch_job_data(repo: str, run_id: int) -> dict[str, Any]:
    """Fetch GitHub Actions job data."""
    logger.info("Fetching job data", repo=repo, run_id=run_id)
    stdout, stderr, code = run_command(["gh", "api", f"repos/{repo}/actions/runs/{run_id}"])
    if code != 0:
        raise RuntimeError(f"Failed to fetch job data: {stderr}")
    return json.loads(stdout)


def clone_repo_at_ref(repo_url: str, ref: str, target_dir: Path) -> None:
    """Clone a repository at a specific ref (branch/commit)."""
    logger.info("Cloning repository", repo_url=repo_url, ref=ref, target_dir=str(target_dir))
    _, stderr, code = run_command(
        ["git", "clone", "--depth", "1", "--branch", ref, repo_url, str(target_dir)]
    )
    if code != 0:
        # Try without --branch if it's a commit SHA
        logger.info("Trying full clone with checkout", ref=ref)
        _, stderr, code = run_command(["git", "clone", repo_url, str(target_dir)])
        if code != 0:
            raise RuntimeError(f"Failed to clone repository: {stderr}")
        _, stderr, code = run_command(["git", "checkout", ref], cwd=target_dir)
        if code != 0:
            raise RuntimeError(f"Failed to checkout ref {ref}: {stderr}")


def create_submission_structure(
    repo_dir: Path,
    paper_id: str,
    output_dir: Path,
    pr_data: dict[str, Any],
    job_data: dict[str, Any],
    comment_data: dict[str, Any],
) -> None:
    """Create a submission structure compatible with run_judge.py."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-UTC")
    submission_dir = output_dir / timestamp

    logger.info("Creating submission structure", submission_dir=str(submission_dir))
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Create the main submission directory
    submission_path = submission_dir / "submission"
    shutil.copytree(repo_dir, submission_path, ignore=shutil.ignore_patterns(".git"))

    # Create metadata.json
    metadata = {
        "paper_id": paper_id,
        "source": "cosci_pr",
        "pr_number": pr_data["number"],
        "pr_url": pr_data["html_url"],
        "pr_title": pr_data["title"],
        "head_sha": pr_data["head"]["sha"],
        "head_ref": pr_data["head"]["ref"],
        "base_sha": pr_data["base"]["sha"],
        "base_ref": pr_data["base"]["ref"],
        "job_id": job_data["id"],
        "job_url": job_data["html_url"],
        "job_status": job_data["status"],
        "job_conclusion": job_data["conclusion"],
        "comment_id": comment_data["id"],
        "comment_url": comment_data["html_url"],
        "created_at": timestamp,
    }

    with open(submission_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Create a simple log.json
    log_data = {
        "timestamp": timestamp,
        "source": "cosci_pr",
        "extraction_method": "run_cosci_submission.py",
    }

    with open(submission_dir / "log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    # Save Claude's comment as a reference
    with open(submission_dir / "claude_comment.txt", "w") as f:
        f.write(comment_data.get("body", ""))

    logger.info("Submission structure created", path=str(submission_dir))
    print(f"\nSubmission created at: {submission_dir}")
    print("\nTo evaluate this submission, run:")
    print(
        f"python paperbench/scripts/run_judge.py --submission-path {submission_path} --paper-id {paper_id} --judge simple --model gpt-5-nano --out-dir {submission_dir}"
    )


async def main(
    repo: str,
    pr_number: int,
    paper_id: str,
    output_dir: Path,
) -> None:
    """Main function to fetch PR data and create submission."""
    # Fetch PR data
    pr_data = fetch_pr_data(repo, pr_number)
    logger.info("PR fetched", title=pr_data["title"], state=pr_data["state"])

    # Fetch comments
    comments = fetch_pr_comments(repo, pr_number)
    claude_comment = find_claude_comment(comments)

    if not claude_comment:
        raise RuntimeError("Could not find Claude's comment with job information")

    logger.info("Found Claude comment", comment_id=claude_comment["id"])

    # Extract job URL and fetch job data
    job_url = extract_job_url_from_comment(claude_comment)
    if not job_url:
        raise RuntimeError("Could not extract job URL from Claude's comment")

    # Parse run ID from job URL
    import re

    match = re.search(r"/actions/runs/(\d+)", job_url)
    if not match:
        raise RuntimeError(f"Could not parse run ID from job URL: {job_url}")

    run_id = int(match.group(1))
    job_data = fetch_job_data(repo, run_id)

    logger.info(
        "Job fetched",
        job_id=job_data["id"],
        status=job_data["status"],
        conclusion=job_data["conclusion"],
    )

    # Clone the repository at the PR branch
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_url = f"https://github.com/{repo}.git"
        head_ref = pr_data["head"]["ref"]

        clone_repo_at_ref(repo_url, head_ref, repo_dir)

        # Create submission structure
        create_submission_structure(
            repo_dir=repo_dir,
            paper_id=paper_id,
            output_dir=output_dir,
            pr_data=pr_data,
            job_data=job_data,
            comment_data=claude_comment,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a submission from a CoSci GitHub PR for paperbench evaluation."
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="GitHub repository in format owner/repo (e.g., josancamon19/paperbench-rice-paper-replication-study)",
    )
    parser.add_argument(
        "--pr-number",
        type=int,
        required=True,
        help="Pull request number",
    )
    parser.add_argument(
        "--paper-id",
        type=str,
        required=True,
        help="Paper identifier (e.g., 'rice', 'semantic-self-consistency')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Output directory for submission (default: runs/)",
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            repo=args.repo,
            pr_number=args.pr_number,
            paper_id=args.paper_id,
            output_dir=args.output_dir,
        )
    )

# Branch Protection Setup

This document explains how to enable branch protection rules to ensure all PRs pass tests before merging.

## Automated Testing

The repository includes a GitHub Actions workflow (`.github/workflows/test.yml`) that automatically runs the full test suite on every pull request to the `main` branch.

## Setting Up Branch Protection Rules

To block PRs that don't pass tests, configure branch protection in GitHub:

### Steps:

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Branches**
3. Under "Branch protection rules", click **Add rule**
4. Configure the following:

   **Branch name pattern:** `main`

   **Protect matching branches:**
   - ☑ Require a pull request before merging
     - ☑ Require approvals: 1 (optional, recommended for team projects)
   - ☑ Require status checks to pass before merging
     - ☑ Require branches to be up to date before merging
     - **Status checks that are required:**
       - Select `test` (this will appear after the first workflow run)
   - ☑ Do not allow bypassing the above settings (recommended)

5. Click **Create** or **Save changes**

### What This Does:

- **Blocks merging** if the `test` workflow fails
- **Requires** all tests to pass before a PR can be merged
- **Ensures** code quality and prevents broken code from entering main branch
- **Automatically runs** on every push to a PR branch

## Workflow Details

The CI/CD workflow (`.github/workflows/test.yml`):

- Triggers on: Pull requests to `main` and pushes to `main`
- Sets up: PostgreSQL, Redis, Elasticsearch, ChromaDB, and API services
- Runs: Full test suite using `pytest`
- Reports: Test results and shows logs on failure
- Timeout: 15 minutes maximum

## Local Testing

Before pushing, you can run the same tests locally:

```bash
# Run all tests
docker compose exec api pytest

# Run with verbose output
docker compose exec api pytest -v

# Run specific test file
docker compose exec api pytest tests/test_detection/test_clustering.py -v
```

## Troubleshooting

### Workflow Not Showing in Status Checks

If the `test` workflow doesn't appear in the status checks list:
1. Merge this PR first
2. Create a new test PR
3. The workflow will run and then appear in the status checks dropdown
4. Go back to Settings → Branches and add it to required checks

### Tests Failing in CI but Passing Locally

Common issues:
- **Environment differences:** Check the `.env` file creation in the workflow
- **Service startup timing:** The workflow waits for services, but you may need to increase sleep times
- **Resource constraints:** GitHub Actions runners have limited resources

Check the workflow logs in the "Actions" tab for detailed error messages.

## Best Practices

1. **Always run tests locally** before pushing
2. **Keep tests fast** - the workflow has a 15-minute timeout
3. **Fix broken tests immediately** - don't let them accumulate
4. **Review test logs** in failed workflow runs for debugging

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [Required Status Checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches#require-status-checks-before-merging)

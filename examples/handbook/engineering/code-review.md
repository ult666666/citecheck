---
fictional: true
notice: Written for the citecheck demo corpus. Larkmoor Analytics is not a real company, and this is not a real policy.
---
# Code Review

## Approval rules

Every change merged into the main branch needs at least one approving review. Changes to authentication, billing or data-retention code need two approvals, and at least one of them must come from the team that owns the code.

## Size of changes

Keep pull requests under 400 changed lines where possible. Split larger work into a sequence of smaller pull requests.

## Review turnaround

Reviewers respond to a review request within 1 business day, even if only to say when they will review it.

## Merging

All automated checks must pass before merging. Authors may not merge their own change without an approval, except for an emergency fix during an active SEV1 or SEV2 incident, which must be reviewed afterward.

---
fictional: true
notice: Written for the citecheck demo corpus. Larkmoor Analytics is not a real company, and this is not a real policy.
---
# Deployments

## Deployment windows

Production deployments happen Monday to Thursday between 09:00 and 16:00 US Eastern Time. There are no deployments on Fridays or on the day before a company holiday, except fixes for SEV1 or SEV2 incidents.

## Change freeze

A change freeze applies during the last two weeks of December. Only incident fixes are deployed during the freeze.

## Canary releases

Every release goes to 5 percent of traffic first and stays there for 30 minutes. If error rates stay normal, the release continues to all traffic.

## Rollbacks

Any engineer may roll back a release at any time without asking for approval. Explain the rollback in the team channel afterward.

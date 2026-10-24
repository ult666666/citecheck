---
fictional: true
notice: Written for the citecheck demo corpus. Larkmoor Analytics is not a real company, and this is not a real policy.
---
# API Style Guide

## Resources and fields

Public APIs use JSON over HTTPS. Resource names are plural nouns, such as /invoices, and field names use snake_case.

## Versioning

The major version is part of the URL, as in /v1/invoices. Breaking changes require a new major version. Deprecations are announced at least 6 months before an endpoint is removed.

## Pagination

List endpoints use cursor-based pagination. The default page size is 50 items and the maximum is 200.

## Errors

Error responses return a JSON body with a machine-readable code, a human-readable message and the request_id of the failed request.

## Timestamps

All timestamps use the ISO 8601 format in UTC.

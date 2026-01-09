# Security agent notes

Checklist for changes:
- No secrets in code or config committed
- Dependencies updated and CI green
- Avoid logging raw rows / PII
- Any new file upload surface validated (file types, size limits)

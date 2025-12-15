# Security Policy

## Supported Versions

The following versions of Wan2.1 are currently being supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 2.x     | :white_check_mark: |
| 1.x     | :x:                |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of Wan2.1 seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Where to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via:

1. **GitHub Security Advisory**: Use the [Security tab](https://github.com/Kuaishou/Wan2.1/security/advisories/new) to privately report vulnerabilities
2. **Email**: Contact the project maintainers (if email is provided in project documentation)

### What to Include

Please include the following information in your report:

- **Description**: A clear description of the vulnerability
- **Type**: The type of vulnerability (e.g., remote code execution, information disclosure, denial of service)
- **Impact**: The potential impact of the vulnerability
- **Steps to Reproduce**: Detailed steps to reproduce the vulnerability
- **Proof of Concept**: If possible, include a minimal proof of concept
- **Affected Versions**: Which versions of Wan2.1 are affected
- **Suggested Fix**: If you have suggestions for fixing the vulnerability

### Example Report

```
**Description**: Arbitrary code execution through malicious model checkpoint

**Type**: Remote Code Execution (RCE)

**Impact**: An attacker could execute arbitrary Python code by crafting a
malicious model checkpoint file.

**Steps to Reproduce**:
1. Create a malicious checkpoint using pickle
2. Load the checkpoint using torch.load()
3. Code executes during unpickling

**Affected Versions**: All versions < 2.1.0

**Suggested Fix**: Use weights_only=True in torch.load() calls
```

## Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Varies by severity (see below)

### Severity Levels

| Severity | Response Time | Fix Timeline |
|----------|---------------|--------------|
| Critical | 24 hours     | 1-7 days     |
| High     | 48 hours     | 7-30 days    |
| Medium   | 7 days       | 30-90 days   |
| Low      | 14 days      | Best effort  |

## Security Best Practices

When using Wan2.1, please follow these security best practices:

### 1. Model Checkpoints

- **Only load trusted checkpoints**: Never load model weights from untrusted sources
- **Verify checksums**: Always verify checkpoint checksums before loading
- **Use safe loading**: The library now uses `weights_only=True` for all `torch.load()` calls

### 2. API Keys and Credentials

- **Environment Variables**: Store API keys in environment variables, never in code
- **Key Rotation**: Rotate API keys regularly
- **Minimal Permissions**: Use API keys with minimal required permissions

```bash
# Good
export DASH_API_KEY="your-api-key"

# Bad - never commit keys to version control
DASH_API_KEY = "sk-abc123..."  # Don't do this!
```

### 3. Input Validation

- **Validate file paths**: Always validate user-provided file paths
- **Sanitize inputs**: Sanitize all user inputs before processing
- **Size limits**: Enforce reasonable size limits on input files

### 4. Network Security

- **HTTPS only**: Use HTTPS for all API communications
- **Verify SSL**: Always verify SSL certificates
- **Timeout settings**: Set appropriate timeouts for network requests

### 5. Dependency Management

- **Keep updated**: Regularly update dependencies to get security patches
- **Audit dependencies**: Run `pip audit` to check for known vulnerabilities
- **Pin versions**: Pin dependency versions in production

```bash
# Check for vulnerabilities
pip install pip-audit
pip-audit

# Update dependencies
pip install --upgrade -r requirements.txt
```

### 6. Execution Environment

- **Sandboxing**: Run in isolated environments when processing untrusted inputs
- **Resource limits**: Set memory and computation limits
- **User permissions**: Run with minimal required user permissions

## Known Security Considerations

### 1. Model Checkpoint Loading

**Fixed in v2.1.0**: All `torch.load()` calls now use `weights_only=True` to prevent arbitrary code execution.

**Before v2.1.0**: Loading untrusted model checkpoints could lead to arbitrary code execution through pickle deserialization.

### 2. Temporary Files

**Status**: The library uses `/tmp` for video caching. Ensure proper permissions on temporary directories.

**Mitigation**: Set appropriate permissions on your system's temp directory, or configure a custom cache directory.

### 3. GPU Memory

**Status**: Processing very large videos can consume significant GPU memory, potentially causing denial of service.

**Mitigation**: Implement resource limits and input validation in production environments.

### 4. API Integration

**Status**: Integration with external APIs (DashScope) requires proper API key management.

**Mitigation**: Always use environment variables for API keys and never commit them to version control.

## Security Updates

Security updates will be released as:

- **Patch releases** for critical and high severity issues
- **Minor releases** for medium severity issues
- **Major releases** for issues requiring breaking changes

Subscribe to:
- GitHub Security Advisories
- Release notifications
- Project announcements

## Disclosure Policy

- **Private Disclosure**: We practice responsible disclosure
- **Coordinated Release**: Security fixes are coordinated with affected parties
- **Public Disclosure**: After a fix is released, we publish a security advisory
- **CVE Assignment**: We request CVEs for significant vulnerabilities

## Bug Bounty Program

We currently do not have a formal bug bounty program. However, we deeply appreciate security researchers who report vulnerabilities responsibly and will acknowledge their contributions in:

- Security advisories
- Release notes
- Project documentation

## Security Checklist for Developers

When contributing to Wan2.1, please ensure:

- [ ] No hardcoded credentials or API keys
- [ ] Input validation for all user-provided data
- [ ] Proper error handling without information leakage
- [ ] Safe deserialization practices (`weights_only=True`)
- [ ] No use of dangerous functions (`eval`, `exec`)
- [ ] Dependency security scan passes
- [ ] Security tests included for new features

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [PyTorch Security](https://pytorch.org/docs/stable/notes/security.html)
- [GitHub Security Best Practices](https://docs.github.com/en/code-security)

## Questions?

For security-related questions that are not sensitive enough to require private disclosure, you may:

- Open a GitHub Discussion
- Contact maintainers through official channels

For all other security matters, please use the private reporting methods described above.

## Acknowledgments

We thank the security researchers and community members who help keep Wan2.1 secure.

---

Last updated: 2025-01-19

# Security Policy

## Supported Versions

The Manifold project provides security updates for the following versions. Users running older versions are encouraged to upgrade to receive security fixes and benefit from improvements in the latest release.

| Version | Supported |
|---------|-----------|
| 2.6.x   | Yes       |
| 2.5.x   | Support may be limited |
| < 2.5   | No        |

Security updates are prioritized for the latest minor version (2.6.x) and the most recent major version. Critical security patches may be backported to older versions when feasible, depending on the nature of the vulnerability and the scope of required changes.

## Reporting a Vulnerability

The Manifold Laboratory takes security seriously and appreciates responsible disclosure of vulnerabilities. If you discover a security vulnerability in Manifold, please report it through our responsible disclosure process.

### How to Report

To report a security vulnerability, please follow these steps:

First, do not disclose the vulnerability publicly. Public disclosure before a fix is available puts users at risk and undermines the responsible disclosure process. Instead, gather all relevant information about the vulnerability including the affected components, steps to reproduce the issue, any proof-of-concept code, and your assessment of the potential impact.

Next, send a detailed report to the security team through GitHub's private vulnerability reporting feature. This feature allows you to submit vulnerability reports directly to project maintainers while keeping the report private. Include all information you have gathered about the vulnerability and any suggested remediation approaches if you have them.

### What to Include

A good vulnerability report includes:

A clear description of the vulnerability including what it affects and how it can be exploited. Detailed steps to reproduce the vulnerability, preferably with minimal steps that demonstrate the issue. Any proof-of-concept code or scripts that demonstrate the vulnerability in action. Your assessment of the severity and potential impact of the vulnerability, including affected components and potential attack vectors. Information about any known mitigations or workarounds. Your contact information for follow-up questions about the report.

### Response Process

Upon receiving a vulnerability report, the security team will:

Acknowledge receipt of your report within 48 hours. This acknowledgment confirms that your report has been received and is being reviewed.

Evaluate the report to confirm the vulnerability and assess its severity. The security team will determine the impact and scope of the vulnerability and develop an appropriate response plan.

Develop and test a fix for the vulnerability. This process involves understanding the root cause, implementing a patch, and verifying that the fix addresses the vulnerability without introducing regressions.

Coordinate the release of the fix. The security team will determine the appropriate timing for disclosure and will work with you on coordinating public disclosure.

You will be kept informed throughout the process of the status of your report and the timeline for resolution. After a fix is released, you will be credited for your responsible disclosure unless you prefer to remain anonymous.

## Security Best Practices

While we work to secure Manifold itself, users should also follow security best practices when deploying and operating Manifold:

### Environment Security

Ensure that the systems running Manifold are properly secured including operating system updates, firewall configuration, and access controls. Use secure configurations for any associated databases or storage systems. Protect API keys, tokens, and other credentials used in production deployments.

### Model Security

When deploying trained models, be aware of potential adversarial inputs that could cause unexpected behavior. Validate and sanitize inputs to models when operating in untrusted environments. Consider the implications of model outputs in security-sensitive applications.

### Dependency Management

Keep Manifold and its dependencies up to date with the latest security patches. Monitor announcements from PyTorch and other dependencies for security updates. Use dependency scanning tools to identify and address vulnerabilities in transitive dependencies.

### Network Security

If running Manifold as a service, apply appropriate network security controls including TLS for data in transit and network segmentation. Monitor for unusual activity that could indicate exploitation attempts.

## Scope

This security policy applies to the Manifold codebase and infrastructure. This includes:

- The main Manifold repository at github.com/Manifold-Laboratory/manifold
- Official documentation in the docs/ directory
- The project website and any associated infrastructure
- Official release artifacts and packages

This policy does not apply to third-party integrations, deployments, or applications built on top of Manifold. Operators of such systems are responsible for their own security configurations and updates.

## Contact

For security questions not related to vulnerability reporting, or for inquiries about security practices in Manifold deployments, please open an issue on GitHub with the "security" label.

For sensitive security matters that should not be discussed publicly, please contact the Manifold Laboratory through GitHub's private vulnerability reporting feature.

## Acknowledgments

We thank the security researchers and community members who help improve Manifold by responsibly reporting vulnerabilities. Your contributions help protect users of the project and improve the overall security posture of the software ecosystem.

---

*This security policy is subject to updates as the project evolves. The latest version is always available in the SECURITY.md file in the repository.*

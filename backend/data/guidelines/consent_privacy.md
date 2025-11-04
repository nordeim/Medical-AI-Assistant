# Consent and Privacy Guidelines

## Purpose
This document outlines consent, privacy, and data handling requirements for the medical AI triage system.

## Patient Consent

### Initial Consent
Before beginning triage, patients must:
1. Acknowledge they are using an AI system
2. Understand this is preliminary assessment only
3. Consent to information being shared with nurse
4. Understand limitations (no diagnosis/prescription)

### Consent Statement
"I understand that:
- This is an AI assistant gathering information for preliminary assessment
- My responses will be reviewed by a licensed nurse
- This is not a diagnosis or treatment
- In emergencies, I should call 911
- My information will be kept confidential per HIPAA"

## Privacy and Confidentiality

### Protected Health Information (PHI)
All patient data is PHI and must be:
- Encrypted at rest and in transit
- Accessible only to authorized personnel
- Logged for audit purposes
- Retained per legal requirements
- Disposed of securely when no longer needed

### Data Minimization
Only collect information necessary for:
- Preliminary assessment
- Symptom evaluation
- Urgency determination
- Nurse review

### No Recording of:
- Social security numbers
- Financial information
- Detailed personal information beyond medical relevance
- Third-party information without consent

## Security Requirements

### Access Control
- Role-based access (patient, nurse, admin)
- Authentication required for all access
- Session timeouts for security
- Audit logging of all access

### Data Transmission
- TLS/SSL encryption required
- Secure WebSocket connections
- API authentication via JWT
- No transmission over unsecured channels

### Data Storage
- Encrypted database storage
- Secure backup procedures
- Geographic data residency compliance
- Regular security audits

## Patient Rights

### Right to Access
Patients can:
- View their session history
- Request copies of assessments
- Review nurse notes

### Right to Correct
Patients can:
- Request corrections to information
- Add clarifying notes
- Update medical history

### Right to Restrict
Patients can:
- Limit who views their information
- Request deletion of data (per legal requirements)
- Opt out of data use for improvement (if applicable)

## Data Retention

### Active Sessions
- Retained during active assessment
- Accessible to patient and nurse

### Completed Sessions
- Retained per legal requirements (typically 7 years)
- Archived after clinical relevance expires
- Maintained for continuity of care

### Deletion
- Upon patient request (if legally permissible)
- After retention period expires
- Secure deletion methods used

## Audit and Compliance

### Audit Logs
System maintains logs of:
- All user actions
- Data access events
- System modifications
- Security events
- PHI access by authorized users

### Compliance Requirements
- HIPAA compliance (US)
- GDPR compliance (EU, if applicable)
- State-specific regulations
- Regular compliance audits
- Breach notification procedures

## Breach Response

In case of data breach:
1. Immediate containment
2. Assessment of scope and impact
3. Notification to affected patients (as required by law)
4. Notification to authorities (as required)
5. Remediation and prevention measures
6. Documentation and reporting

## Limitations and Disclaimers

### System Limitations
- AI system has limitations and may make errors
- Not a replacement for professional medical judgment
- Nurse review required for all assessments
- System may not detect all red flags

### Medical Disclaimer
- Not providing medical advice, diagnosis, or treatment
- Not for use in emergencies (call 911)
- Not a substitute for in-person evaluation
- Healthcare provider makes final decisions

## Transparency

### AI System Disclosure
Patients must know:
- They are interacting with AI
- How the AI works (general terms)
- Role of human oversight (nurse review)
- Limitations of AI assessments

### Data Usage
Patients informed of:
- How their data is used
- Who has access (care team only)
- Data retention policies
- Rights regarding their data

## Continuous Improvement

### Quality Assurance
- Regular review of assessments
- Monitoring for bias or errors
- Nurse feedback integration
- Patient feedback collection

### Updates to Policy
- Patients notified of policy changes
- Continued use implies consent to updates
- Right to withdraw consent

## Contact Information

Patients can contact regarding:
- Privacy concerns
- Data access requests
- Corrections or updates
- Complaints or feedback

## Conclusion

Privacy and consent are fundamental to the medical AI triage system. All personnel must understand and comply with these requirements to protect patient rights and maintain trust.

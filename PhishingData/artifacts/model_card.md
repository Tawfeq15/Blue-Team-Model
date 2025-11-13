# Model Card: Phishing Detection System

## Model Details
- **Model Type**: LightGBM
- **Version**: 1.0.0
- **Date Trained**: 2025-11-09T21:51:46.793061
- **Training Time**: 5711.37 seconds
- **Optimal Threshold**: 0.500

## Intended Use
- **Primary**: Detect phishing in emails/SMS/messages.
- **Users**: Security teams, email providers, enterprises.
- **Out-of-Scope**: Non-phishing cyber threats.

## Performance (Test)
- **Accuracy**: 0.9968
- **Precision**: 0.9980
- **Recall**: 0.9944
- **F1**: 0.9962
- **ROC-AUC**: 0.9995
- **PR-AUC**: 0.9994
- **Recall@FPR≤0.3%**: 0.9948

## Limitations
- New attack styles may degrade performance.
- Retrain periodically (≈ every 30 days).
- Performance may drop on non-English content.
- Susceptible to sophisticated adversarial attacks.

## Governance
- **Drift Monitoring**: Weekly
- **Performance Review**: Monthly
- **Security Audit**: Quarterly
- **Retraining**: 30 days

## Contact
- **Team**: Security ML Team
- **Email**: security-ml@company.com
    
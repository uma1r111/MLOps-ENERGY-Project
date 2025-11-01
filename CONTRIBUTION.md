# 🧩 CONTRIBUTION.md  
### MLOps Energy Forecasting System – Milestone 1  
---

## 👥 Contributors

| Name | ERP ID |
|------|---------|
| **Sameed Ahmad** | 26956 |
| **Izma Khan** | 26926 |
| **Shaikh Muhammad Umair** | 26409 |
| **Abdullah Rehman** | 27074 |


---

## 🧠 Task Allocation & Contributions

| Member | Role / Area | Detailed Contributions |
|---------|--------------|------------------------|
| **Sameed Ahmad** | 🧱 *Infrastructure & CI/CD* | Set up project structure, Makefile, Dockerfile, and GitHub Actions workflow for build, test, and deployment. Configured pre-commit hooks and security checks. |

| **Izma Khan** | 📊 *Data Pipeline & Monitoring* | Handled data preparation and feature engineering. Implemented Evidently AI dashboard monitoring. Drafted and formatted the main **README.md**. |

| **Shaikh Muhammad Umair** | 🤖 *Model & MLflow* | Developed and trained the forecasting model. Configured MLflow tracking and registered model v1 with S3 backend. Developed the **BentoML** inference service with `/predict`, `/health`, and `/docs` endpoints. |

| **Abdullah Rehman** | 🌐 *API & Cloud Deployment* | AWS EC2 deployment, and S3 integration. Configured pip-audit vulnerability scans, and code compliance checks. |


---

## 🌿 Branch Naming Convention

| Prefix | Example | Purpose |
|---------|----------|----------|
| `main` | `main`| Production-ready branch.
| `feature/` | `feat/monitoring` | Monitoring scripts. |
| `infra/` | `infra/docker-setup` | Dockerfile and container configuration. |
| `test/` | `test/add-unit-tests` | Adds and maintains unit tests. |
| `security/` | `sec/dependency-audit` | Security audits and vulnerability scanning.|
---

### 🧾 Notes
- All contributors followed the branch-naming and pull request review workflow.  
- Each task was validated via CI before merging into `main`.  


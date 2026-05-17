import argparse
import json
from pathlib import Path


EXAMPLES = [
    {
        "instruction": (
            "Given this job description, generate a tailored 1-line resume bio and "
            "6 technical skill headers with relevant skills for each.\n\n"
            "Job Description: ML Engineer at a fintech startup. Requirements: Python, "
            "TensorFlow, fraud detection, real-time ML pipelines, AWS SageMaker, "
            "feature engineering, A/B testing, SQL."
        ),
        "output": (
            "Bio: Engineer with production ML systems experience delivering $1.5M USD "
            "in business impact through scalable data pipelines and predictive modeling.\n\n"
            "Skills:\n"
            "Machine Learning & Modeling: XGBoost, Feature Engineering, Hyperparameter Tuning, Model Evaluation, A/B Testing, Fraud Detection\n"
            "Cloud & MLOps: AWS SageMaker, S3, Lambda, EC2, CloudWatch, ML Pipelines\n"
            "Data Engineering: SQL, ETL Pipelines, AWS Glue, Redshift, Data Warehousing\n"
            "Programming: Python, Pandas, NumPy, Scikit-learn, TensorFlow\n"
            "Real-Time Systems: Streaming Pipelines, Low-Latency Inference, API Development, FastAPI\n"
            "Analytics & Visualization: Tableau, EDA, Hypothesis Testing, Statistical Analysis"
        ),
    },
    {
        "instruction": (
            "Given this job description, generate a tailored 1-line resume bio and "
            "6 technical skill headers with relevant skills for each.\n\n"
            "Job Description: Data Scientist at a healthcare company. Requirements: "
            "Python, R, NLP, clinical data, predictive modeling, Tableau, SQL, "
            "statistical analysis, experiment design."
        ),
        "output": (
            "Bio: Engineer with production ML systems experience delivering $1.5M USD "
            "in business impact through predictive modeling and data-driven insights.\n\n"
            "Skills:\n"
            "Statistical Modeling & ML: Predictive Modeling, XGBoost, Regression, Hypothesis Testing, Experiment Design, EDA\n"
            "NLP & Text Analytics: Text Classification, Named Entity Recognition, Tokenization, Sentiment Analysis, spaCy\n"
            "Data Analysis: Python, Pandas, NumPy, Scikit-learn, R, Statistical Analysis\n"
            "Databases & SQL: SQL, Redshift, PostgreSQL, Data Warehousing, ETL Pipelines\n"
            "Visualization & Reporting: Tableau, Matplotlib, Seaborn, Dashboard Design\n"
            "Cloud & Infrastructure: AWS SageMaker, S3, EC2, Lambda, CloudWatch"
        ),
    },
    {
        "instruction": (
            "Given this job description, generate a tailored 1-line resume bio and "
            "6 technical skill headers with relevant skills for each.\n\n"
            "Job Description: AI Engineer building multi-agent systems. Requirements: "
            "LangChain, LangGraph, OpenAI API, vector databases, RAG pipelines, "
            "FastAPI, Docker, Kubernetes, PostgreSQL."
        ),
        "output": (
            "Bio: Engineer with production ML and multi-agent systems experience "
            "delivering $1.5M USD in business impact through scalable AI architectures.\n\n"
            "Skills:\n"
            "LLM & Agent Frameworks: LangChain, LangGraph, OpenAI API, GPT-4, Multi-Agent Orchestration, Prompt Engineering\n"
            "RAG & Vector Databases: Qdrant, ChromaDB, Retrieval-Augmented Generation, Embedding Models, Semantic Search\n"
            "Backend & APIs: FastAPI, REST APIs, Python, PostgreSQL, Redis\n"
            "Cloud & DevOps: Docker, Kubernetes, AWS, GCP Cloud Run, CI/CD, GitHub Actions\n"
            "Data Engineering: ETL Pipelines, SQL, Data Modeling, AWS Glue, S3\n"
            "Testing & Monitoring: Pytest, Unit Testing, CloudWatch, Logging, Observability"
        ),
    },
    {
        "instruction": (
            "Given this job description, generate a tailored 1-line resume bio and "
            "6 technical skill headers with relevant skills for each.\n\n"
            "Job Description: Backend ML Platform Engineer. Requirements: Python, "
            "FastAPI, Kubernetes, model serving, MLflow, Docker, PostgreSQL, Redis, "
            "CI/CD, monitoring."
        ),
        "output": (
            "Bio: Engineer with production ML systems experience delivering $1.5M USD "
            "in business impact through scalable model serving and data platforms.\n\n"
            "Skills:\n"
            "Model Serving & MLOps: MLflow, Model Registry, Batch Inference, Online Inference, Feature Stores, Model Monitoring\n"
            "Backend Engineering: FastAPI, REST APIs, Python, Pydantic, PostgreSQL, Redis\n"
            "Cloud & Containers: Docker, Kubernetes, AWS ECS, EKS, S3, CloudWatch\n"
            "Data Systems: SQL, ETL Pipelines, Data Modeling, Redshift, AWS Glue\n"
            "CI/CD & Testing: GitHub Actions, Pytest, Unit Testing, Integration Testing, Release Automation\n"
            "Observability: Logging, Metrics, Tracing, Alerting, Performance Profiling"
        ),
    },
    {
        "instruction": (
            "Given this job description, generate a tailored 1-line resume bio and "
            "6 technical skill headers with relevant skills for each.\n\n"
            "Job Description: Analytics Engineer for a SaaS company. Requirements: "
            "dbt, Snowflake, SQL, Python, Looker, data modeling, metrics layers, "
            "ETL, stakeholder dashboards."
        ),
        "output": (
            "Bio: Engineer with production analytics and ML systems experience delivering "
            "$1.5M USD in business impact through trusted data models and decision workflows.\n\n"
            "Skills:\n"
            "Analytics Engineering: dbt, Dimensional Modeling, Metrics Layers, Semantic Models, Data Contracts\n"
            "Warehousing & SQL: Snowflake, SQL, Redshift, Query Optimization, Data Warehousing\n"
            "Data Pipelines: ETL Pipelines, ELT Workflows, Airflow, AWS Glue, Data Quality Checks\n"
            "Programming: Python, Pandas, NumPy, Jupyter, API Integrations\n"
            "BI & Visualization: Looker, Tableau, Dashboard Design, KPI Reporting, Cohort Analysis\n"
            "Testing & Governance: dbt Tests, Schema Validation, Lineage, Documentation, Access Controls"
        ),
    },
    {
        "instruction": (
            "Given this job description, generate a tailored 1-line resume bio and "
            "6 technical skill headers with relevant skills for each.\n\n"
            "Job Description: Computer Vision Engineer. Requirements: PyTorch, OpenCV, "
            "object detection, segmentation, model optimization, CUDA, edge deployment, "
            "Python, AWS."
        ),
        "output": (
            "Bio: Engineer with production ML systems experience delivering $1.5M USD "
            "in business impact through optimized computer vision and inference pipelines.\n\n"
            "Skills:\n"
            "Computer Vision: Object Detection, Image Segmentation, OpenCV, Feature Extraction, Data Augmentation\n"
            "Deep Learning: PyTorch, CNNs, Transformers, Transfer Learning, Model Evaluation\n"
            "Model Optimization: Quantization, ONNX, TensorRT, CUDA, Latency Profiling\n"
            "Edge & Serving: Batch Inference, Real-Time Inference, FastAPI, Docker, Edge Deployment\n"
            "Cloud & Data: AWS S3, EC2, SageMaker, Dataset Versioning, Annotation Pipelines\n"
            "Programming & Testing: Python, NumPy, Pytest, Experiment Tracking, CI/CD"
        ),
    },
    {
        "instruction": (
            "Given this job description, generate a tailored 1-line resume bio and "
            "6 technical skill headers with relevant skills for each.\n\n"
            "Job Description: Data Engineer. Requirements: Spark, Airflow, Python, "
            "SQL, AWS Glue, S3, Redshift, streaming data, data quality, Terraform."
        ),
        "output": (
            "Bio: Engineer with production data and ML systems experience delivering "
            "$1.5M USD in business impact through reliable pipelines and analytics platforms.\n\n"
            "Skills:\n"
            "Data Engineering: Spark, Airflow, ETL Pipelines, ELT Workflows, Data Quality, Data Lineage\n"
            "Cloud Data Platforms: AWS Glue, S3, Redshift, Lambda, CloudWatch, IAM\n"
            "Streaming Systems: Kafka, Kinesis, Streaming Pipelines, Event Processing, Backfills\n"
            "Databases & SQL: SQL, PostgreSQL, Data Warehousing, Query Optimization, Data Modeling\n"
            "Infrastructure: Terraform, Docker, CI/CD, GitHub Actions, Environment Management\n"
            "Programming: Python, Pandas, PySpark, Unit Testing, Logging"
        ),
    },
    {
        "instruction": (
            "Given this job description, generate a tailored 1-line resume bio and "
            "6 technical skill headers with relevant skills for each.\n\n"
            "Job Description: Quantitative Analyst. Requirements: Python, time-series "
            "analysis, portfolio analytics, factor models, risk modeling, SQL, pandas, "
            "statistical inference."
        ),
        "output": (
            "Bio: Engineer with production analytics and ML systems experience delivering "
            "$1.5M USD in business impact through predictive modeling and financial decision systems.\n\n"
            "Skills:\n"
            "Quantitative Modeling: Time-Series Analysis, Factor Models, Risk Modeling, Portfolio Analytics, Backtesting\n"
            "Statistical Analysis: Hypothesis Testing, Regression, Statistical Inference, Forecasting, Experiment Design\n"
            "Financial Data: Market Data Pipelines, Performance Attribution, Volatility Modeling, Scenario Analysis\n"
            "Programming: Python, Pandas, NumPy, SciPy, Scikit-learn, Jupyter\n"
            "Databases & SQL: SQL, PostgreSQL, Redshift, Data Warehousing, Query Optimization\n"
            "Visualization & Reporting: Tableau, Matplotlib, Seaborn, KPI Dashboards, Analytical Reporting"
        ),
    },
    {
        "instruction": (
            "Given this job description, generate a tailored 1-line resume bio and "
            "6 technical skill headers with relevant skills for each.\n\n"
            "Job Description: Generative AI Product Engineer. Requirements: LLM APIs, "
            "RAG, prompt evaluation, TypeScript, Python, FastAPI, Postgres, vector search, "
            "production monitoring."
        ),
        "output": (
            "Bio: Engineer with production generative AI and ML systems experience "
            "delivering $1.5M USD in business impact through scalable AI applications.\n\n"
            "Skills:\n"
            "Generative AI: LLM APIs, Prompt Engineering, Prompt Evaluation, Function Calling, Structured Outputs\n"
            "RAG Systems: Vector Search, Embeddings, Retrieval Evaluation, Reranking, Knowledge Base Pipelines\n"
            "Backend Engineering: Python, FastAPI, PostgreSQL, Redis, REST APIs, Async Workflows\n"
            "Frontend & Product: TypeScript, React, API Integration, State Management, Product Analytics\n"
            "MLOps & Monitoring: Observability, Logging, Eval Harnesses, Model Monitoring, CI/CD\n"
            "Data Engineering: ETL Pipelines, SQL, Data Modeling, Dataset Curation, Experiment Tracking"
        ),
    },
    {
        "instruction": (
            "Given this job description, generate a tailored 1-line resume bio and "
            "6 technical skill headers with relevant skills for each.\n\n"
            "Job Description: Business Intelligence Developer. Requirements: Power BI, "
            "Tableau, SQL, data warehousing, DAX, Python, executive dashboards, ETL, "
            "KPI reporting."
        ),
        "output": (
            "Bio: Engineer with production analytics systems experience delivering "
            "$1.5M USD in business impact through executive dashboards and trusted reporting layers.\n\n"
            "Skills:\n"
            "Business Intelligence: Power BI, Tableau, Executive Dashboards, KPI Reporting, Self-Service Analytics\n"
            "Data Modeling: Star Schemas, Dimensional Modeling, Metrics Definitions, Semantic Layers, DAX\n"
            "Databases & SQL: SQL, PostgreSQL, Redshift, Snowflake, Query Optimization\n"
            "Data Pipelines: ETL Pipelines, AWS Glue, Airflow, Data Quality Checks, Scheduled Reporting\n"
            "Programming: Python, Pandas, NumPy, Jupyter, Automation Scripts\n"
            "Analytics: Cohort Analysis, Funnel Analysis, Forecasting, Variance Analysis, Statistical Reporting"
        ),
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/resume_dataset.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        for example in EXAMPLES:
            handle.write(json.dumps(example) + "\n")

    print(f"Created dataset with {len(EXAMPLES)} examples at {output_path}")
    print("Add 200-500 real JD-to-resume examples before serious training.")


if __name__ == "__main__":
    main()

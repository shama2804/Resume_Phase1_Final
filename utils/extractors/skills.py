import re

# Inline TECH_KEYWORDS (abbreviated below – replace with full list if needed)
TECH_KEYWORDS = [
    # Programming & Languages
    "python", "java", "c++", "c#", "html", "css", "javascript", "typescript", "r", "sql", "bash",

    # Web Dev & Frameworks
    "react", "angular", "vue", "node.js", "flask", "django", "fastapi", "bootstrap", "tailwind",

    # Data Science & ML
    "pandas", "numpy", "scipy", "sklearn", "matplotlib", "seaborn", "tensorflow", "keras", "pytorch",
    "openai", "huggingface", "nltk", "spacy", "gensim", "xgboost", "lightgbm", "mlflow", "optuna",

    # Tools
    "git", "github", "bitbucket", "vscode", "jupyter", "colab", "intellij", "postman", "docker", "kubernetes",

    # BI & Analytics
    "excel", "power bi", "tableau", "looker", "qlikview", "superset", "alteryx",

    # UI/UX & Design
    "figma", "canva", "photoshop", "illustrator", "sketch", "xd", "adobe xd", "webflow", "wix", "wordpress",

    # Databases
    "mysql", "postgresql", "mongodb", "firebase", "sqlite", "oracle", "snowflake", "redshift",

    # DevOps & Cloud
    "aws", "azure", "gcp", "jenkins", "terraform", "ansible", "prometheus", "grafana", "elk", "airflow",

    # Embedded & Electronics
    "arduino", "raspberry pi", "iot", "esp32", "verilog", "vhdl", "proteus", "labview",

    # CAD & Mechanical
    "autocad", "solidworks", "catia", "ansys", "fusion 360", "creo", "hypermesh", "nx",

    # Civil & Architecture
    "revit", "staad pro", "etabs", "sketchup", "qgis", "arcgis", "primavera", "ms project",

    # Healthcare / Life Sciences
    "lims", "bioconductor", "pubmed", "genbank", "biopython", "snapgene", "chemdraw", "zotero",

    # Finance / Business
    "tally", "sap", "quickbooks", "xero", "oracle financials", "zoho books",

    # Marketing / Content / Multimedia
    "mailchimp", "hootsuite", "buffer", "semrush", "notion", "obs", "audacity", "after effects", "premiere pro"
]

def extract_skills(text):
    text_lower = text.lower()
    found_skills = set()

    for kw in TECH_KEYWORDS:
        if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
            found_skills.add(kw.title())  # Consistent case

    return sorted(found_skills)

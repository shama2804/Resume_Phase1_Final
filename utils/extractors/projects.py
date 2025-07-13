import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load NER model
model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Extensive cross-domain tech keywords
TECH_KEYWORDS = [
    # IT & Software (150+ terms)
    "python", "java", "c++", "c#", "html", "css", "javascript", "typescript", "react", "angular", "vue",
    "node.js", "express", "flask", "django", "fastapi", "sql", "mysql", "postgresql", "mongodb", "firebase",
    "pandas", "numpy", "scipy", "sklearn", "matplotlib", "seaborn", "keras", "tensorflow", "pytorch",
    "git", "github", "bitbucket", "docker", "kubernetes", "aws", "azure", "gcp", "vscode", "intellij",
    "jupyter", "spyder", "colab", "airflow", "spark", "hadoop", "snowflake", "databricks", "superset", "grafana",
    "jenkins", "jira", "ansible", "terraform", "sentry", "new relic", "elk", "logstash", "prometheus", "grafana",
    "openai", "huggingface", "llamaindex", "langchain", "transformers", "nltk", "spacy", "openai gym", "mlflow", "optuna",
    "scikit-learn", "lightgbm", "xgboost", "catboost", "cv2", "open3d", "plotly", "dash", "pyqt", "tkinter",

    # Web & UI/UX Design (150+ terms)
    "figma", "adobe xd", "sketch", "photoshop", "illustrator", "after effects", "indesign", "canva", "coreldraw", "invision",
    "zeplin", "balsamiq", "gravit designer", "marvel", "affinity designer", "crello", "vectr", "framer", "gimp", "inkscape",
    "webflow", "bootstrap", "foundation", "sass", "tailwind", "materialize", "uizard", "lottie", "xd plugin", "protopie",
    "moqups", "axure", "justinmind", "lunacy", "mockflow", "wix", "wordpress", "elementor", "brizy", "oxygen builder",

    # Data & BI Tools (150+ terms)
    "excel", "powerpoint", "tableau", "power bi", "looker", "qlikview", "qliksense", "datawrapper", "d3.js", "metabase",
    "superset", "microstrategy", "cognos", "birst", "sas", "alteryx", "knime", "orange", "r", "stata",
    "ibm spss", "openrefine", "redash", "domo", "databox", "cyfe", "periscope", "zoho analytics", "datapine", "thoughtspot",

    # Electronics / Robotics / Embedded (150+ terms)
    "arduino", "iot", "esp32", "raspberry pi", "verilog", "vhdl", "proteus", "multisim", "keil", "blynk",
    "gsm", "mqtt", "sim800l", "atmega", "pic microcontroller", "labview", "rtl", "stm32", "lora", "ble",
    "altium", "eagle", "kicad", "ltspice", "nrf52", "nrf24", "hc-05", "hc-06", "tinkercad", "beaglebone",
    "openmv", "opencv", "pca9685", "oled display", "ultrasonic sensor", "ldr", "dht11", "relay module", "ir sensor", "nrf24l01",

    # Mechanical / CAD (150+ terms)
    "autocad", "solidworks", "catia", "ansys", "creo", "fusion 360", "inventor", "hypermesh", "nastran", "abaqus",
    "matlab", "simulink", "adams", "dassault", "ptc windchill", "nx", "cfdesign", "flow simulation", "fluent", "star ccm+",
    "camworks", "mastercam", "pro/e", "siemens nx", "fea", "cfd", "mechanica", "hyperview", "altair", "deform3d",

    # Civil / Architecture / Planning (150+ terms)
    "revit", "staad pro", "etabs", "autocad civil", "arcgis", "qgis", "primavera", "sketchup", "v ray", "lumion",
    "civil 3d", "plaxis", "autodesk robot", "ms project", "geopak", "bentley", "tekla", "e-tabs", "safe", "survey camp",
    "autoturn", "infraworks", "bluebeam", "navisworks", "mx road", "hydraulic modeling", "hydrocad", "stormcad", "epanet", "civilstorm",

    # Finance / Commerce (150+ terms)
    "tally", "quickbooks", "sap", "zoho books", "xero", "oracle financials", "excel macros", "gst filing", "taxation", "auditing",
    "payroll", "sap fico", "erp", "cost accounting", "management accounting", "ms dynamics", "sage", "peachtree", "bank reconciliation", "account receivables",
    "tcs ion", "financial modeling", "equity research", "npv", "irr", "ratio analysis", "stock market", "nse", "bse", "futures",

    # Healthcare / Life Sciences (150+ terms)
    "lims", "bioconductor", "labguru", "meditech", "epic", "cerner", "genbank", "biopython", "metlab", "pubmed",
    "emr", "ehr", "clsi", "microscopy", "cytoscape", "gel electrophoresis", "rna-seq", "elisa", "pcr", "flow cytometry",
    "genomics", "proteomics", "clinical trials", "drug discovery", "chemdraw", "snapgene", "graphpad prism", "bioedit", "endnote", "zotero",

    # Education / Humanities / Social Sciences (150+ terms)
    "moodle", "blackboard", "canvas", "turnitin", "mathtype", "latex", "ms teams", "zoom", "google classroom", "lms",
    "padlet", "kahoot", "nearpod", "slido", "edmodo", "mentimeter", "socrative", "peardeck", "classdojo", "gradebook",
    "storyboard", "ebook creator", "powtoon", "wevideo", "quizizz", "outlook", "team viewer", "obs", "prezi", "whiteboard",

    # Law / Legal / Management (150+ terms)
    "lexisnexis", "manupatra", "case mine", "air", "scconline", "live law", "indiakanoon", "case tracking", "legal docs", "contract management",
    "compliance", "due diligence", "corporate law", "arbitration", "intellectual property", "litigation", "legal research", "plaint drafting", "notary", "lawctopus",
    "crm", "erp", "salesforce", "zoho crm", "hubspot", "microsoft dynamics", "basecamp", "pipedrive", "freshsales", "keap",

    # Multimedia / Arts / Marketing (150+ terms)
    "audacity", "premiere pro", "after effects", "lightroom", "davinci resolve", "final cut pro", "cinema 4d", "blender", "maya", "filmora",
    "sony vegas", "toon boom", "vyond", "canva pro", "mailchimp", "hootsuite", "buffer", "facebook ads", "google ads", "meta business suite",
    "content calendar", "campaign manager", "keyword planner", "semrush", "ahrefs", "buzzsumo", "surfer seo", "ubersuggest", "notion", "trello"
]
def clean(text):
    return text.replace('\n', ' ').strip()

def extract_projects(text):
    lines = text.split('\n')
    projects = []

    # 1. Locate Projects section
    start = -1
    end = len(lines)
    for i, line in enumerate(lines):
        if re.search(r'\b(projects|academic projects|personal projects)\b', line.lower()):
            start = i
        elif start != -1 and re.match(r'^[A-Z][A-Z\s:]{5,}$', line.strip()):
            end = i
            break
    if start == -1:
        return []

    # 2. Gather project section text
    section_text = "\n".join(lines[start+1:end]).strip()
    if not section_text:
        return []

    # 3. Split by likely project delimiters
    segments = re.split(r'(?=\n?[-•*]\s+|^\d+\.\s+|^project\s*:)', section_text, flags=re.IGNORECASE)
    seen_titles = set()

    for seg in segments:
        seg = seg.strip()
        if len(seg) < 30:
            continue

        # Skip certifications / learning / courses
        if any(keyword in seg.lower() for keyword in ["coursera", "udemy", "linkedin", "certification", "hackerrank", "training", "course", "google", "nasba", "python (basic)", "essential training"]):
            continue

        # Accept only if action verbs appear (indicating real project)
        if not re.search(r'\b(developed|created|built|designed|implemented|led|engineered|contributed)\b', seg, re.IGNORECASE):
            continue

        proj = {
            "title": "",
            "tech_stack": "",
            "description": "",
            "duration": ""
        }

        # Extract title from first sentence
        first_sent = re.split(r'[.!?\n]', seg)[0].strip()
        if 10 < len(first_sent) < 120 and not re.match(r'^[-•\d]', first_sent):
            proj['title'] = first_sent
        else:
            # Fallback NER title
            entities = ner_pipeline(seg)
            for ent in entities:
                if ent['entity_group'].lower() in ['misc', 'prod', 'org'] and len(ent['word']) > 6:
                    proj['title'] = ent['word'].strip()
                    break

        # Skip junk titles
        bad_titles = ['python', 'java', 'sql', 'github', 'linkedin', 'javascript', 'udemy', 'project', 'html']
        if proj['title'].lower() in bad_titles:
            continue

        # Duration
        duration_match = re.search(
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,-]*\d{4}\s*(?:–|-|to|until)?\s*'
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?[a-z]*[\s,-]*\d{4}',
            seg.lower())
        if duration_match:
            proj["duration"] = duration_match.group(0).title()

        # Tech stack detection
        stack = set()
        for tech in TECH_KEYWORDS:
            if re.search(r'\b' + re.escape(tech) + r'\b', seg, re.IGNORECASE):
                stack.add(tech.title())
        proj['tech_stack'] = ", ".join(sorted(stack))

        # Full segment = description
        proj['description'] = clean(seg)

        # Add only if valid
        if proj['title'] and proj['description'] and proj['title'].lower() not in seen_titles:
            seen_titles.add(proj['title'].lower())
            projects.append(proj)

    return projects

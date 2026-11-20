from __future__ import annotations

import hashlib
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from translations import translate, translate_indicator

# A running Streamlit process can retain the pre-upgrade module across reruns.
import image_catalog
if not hasattr(image_catalog, "build_verified_catalog"):
    import importlib
    image_catalog = importlib.reload(image_catalog)
from image_catalog import find_catalog, build_verified_catalog, match_reference, image_bytes

st.set_page_config(page_title=translate(st.session_state.get("language", "en"), "Brand Protection Intelligence"), layout="wide")
ROOT = Path(__file__).parent
NUMERIC = ["price", "seller_rating", "spelling_errors"]
CATEGORICAL = ["category", "contact_info_complete", "return_policy_clear"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "is_counterfeit"
DISCLAIMER = "Educational prototype. AI screening results are experimental and require human verification."


def t(message, **values):
    return translate(st.session_state.get("language", "en"), message, **values)


def sync_sort():
    st.session_state.sort_choice = st.session_state[f"queue_sort_{st.session_state.language}"]


def search_cases_changed():
    st.session_state.case_query = st.session_state.case_search
    st.session_state.queue_page = 1


def reset_queue_filters():
    for name, value in [("priority_filter", "All priorities"), ("category_filter", "All categories"), ("status_filter", "All statuses")]:
        st.session_state[name] = value
        for language in ("en", "ko"):
            st.session_state[f"{name}_{language}"] = value
    st.session_state.case_search = ""
    st.session_state.case_query = ""
    st.session_state.queue_page = 1


def localized_select(name, title, options, page_key=None):
    language = st.session_state.language
    selected = st.session_state.get(name, options[0])
    if selected not in options:
        selected = options[0]
    widget_key = f"{name}_{language}"

    def changed():
        st.session_state[name] = st.session_state[widget_key]
        if page_key:
            st.session_state[page_key] = 1

    return st.selectbox(t(title), options, index=options.index(selected), format_func=lambda option: translate(language, option), key=widget_key, on_change=changed)


def page_for(name, count, page_size=24):
    pages = max(1, (count + page_size - 1) // page_size)
    st.session_state[name] = min(max(1, st.session_state.get(name, 1)), pages)
    widget_key = f"{name}_widget"
    if st.session_state.get(widget_key) != st.session_state[name]:
        st.session_state[widget_key] = st.session_state[name]
    def changed():
        st.session_state[name] = st.session_state[widget_key]
    return st.number_input(t("Page"), min_value=1, max_value=pages, step=1, key=widget_key, on_change=changed)


def shown(value, default="Not available in dataset"):
    return t(default) if value is None or pd.isna(value) or str(value).strip() == "" else str(value)


def normalize_target(series):
    return series.astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})


def prepare(raw):
    df = raw.copy()
    df.columns = df.columns.str.strip().str.lower()
    df = df.replace(r"^\s*$", np.nan, regex=True)
    before = len(df)
    df = df.drop_duplicates()
    exact_duplicates = before - len(df)
    duplicate_ids = 0
    if "product_id" in df:
        # Treat repeated IDs as one case and remove all conflicting records.
        ids = df["product_id"].astype("string").str.strip().replace("", pd.NA)
        repeated = ids.notna() & ids.duplicated(keep=False)
        duplicate_ids = int(repeated.sum())
        df = df.loc[~repeated].copy()
    for name in NUMERIC + ["seller_reviews"]:
        if name in df:
            df[name] = pd.to_numeric(df[name], errors="coerce")
    for name in CATEGORICAL:
        if name in df:
            df[name] = df[name].astype("string").str.strip().str.lower().replace({"": pd.NA}).astype(object).where(df[name].notna(), np.nan)
    return df.reset_index(drop=True), exact_duplicates, duplicate_ids


@st.cache_data(show_spinner=False)
def analyze(data, feature_names):
    raw = pd.read_csv(BytesIO(data))
    df, duplicates, duplicate_ids = prepare(raw)
    features = [x for x in feature_names if x in df and df[x].notna().any()]
    y = normalize_target(df[TARGET]) if TARGET in df else pd.Series(np.nan, index=df.index)
    valid = y.notna()
    counts = y[valid].value_counts()
    info = {"raw": raw, "duplicates": duplicates, "duplicate_ids": duplicate_ids, "features": features,
            "counts": counts.to_dict(), "excluded_labels": int((~valid).sum())}
    if len(features) < 2 or len(counts) != 2 or counts.min() < 5 or valid.sum() < 30:
        return df, None, info
    x = df.loc[valid, features]
    y = y.loc[valid].astype(int)
    train_idx, test_idx = train_test_split(x.index, test_size=0.25, random_state=42, stratify=y)
    numeric = [c for c in features if c in NUMERIC]
    categorical = [c for c in features if c in CATEGORICAL]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("category", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    model = Pipeline([("preprocess", ColumnTransformer(transformers)), ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
    model.fit(x.loc[train_idx], y.loc[train_idx])
    predicted = model.predict(x.loc[test_idx])
    scores = model.predict_proba(x.loc[test_idx])[:, 1]
    info.update(precision=precision_score(y.loc[test_idx], predicted, zero_division=0),
                recall=recall_score(y.loc[test_idx], predicted, zero_division=0),
                matrix=confusion_matrix(y.loc[test_idx], predicted, labels=[0, 1]),
                train_size=len(train_idx), test_size=len(test_idx))
    heldout = df.loc[test_idx].copy()
    heldout["score"] = scores
    return heldout, model, info


def synthetic_data():
    # The fallback is not attributed to Kaggle and is never used to train a model.
    return pd.DataFrame([
        {"category": "Fashion", "brand": "Example Studio", "price": 19, "seller_rating": 2.6, "seller_reviews": 12, "spelling_errors": 3, "contact_info_complete": "false", "return_policy_clear": "false"},
        {"category": "Fashion", "brand": "Example Studio", "price": 45, "seller_rating": 4.4, "seller_reviews": 210, "spelling_errors": 0, "contact_info_complete": "true", "return_policy_clear": "true"},
        {"category": "Luxury Goods", "brand": "Sample Label", "price": 29, "seller_rating": 3.2, "seller_reviews": 32, "spelling_errors": 2, "contact_info_complete": "true", "return_policy_clear": "false"},
        {"category": "Fashion", "brand": "Sample Label", "price": 88, "seller_rating": 4.8, "seller_reviews": 500, "spelling_errors": 0, "contact_info_complete": "true", "return_policy_clear": "true"},
        {"category": "Fashion", "brand": "Example Studio", "price": np.nan, "seller_rating": 3.8, "seller_reviews": 70, "spelling_errors": 1, "contact_info_complete": "false", "return_policy_clear": "true"},
    ])


def indicators(row):
    signals, missing = [], []
    for field, title in [("category", "Product category"), ("price", "Price"), ("seller_rating", "Seller rating"), ("contact_info_complete", "Seller contact completeness")]:
        if field not in row or pd.isna(row[field]):
            missing.append(f"{title} is missing.")
    if str(row.get("contact_info_complete", "")).lower() == "false":
        missing.append("Seller contact information is marked incomplete.")
    if str(row.get("return_policy_clear", "")).lower() == "false":
        signals.append("Return policy is marked unclear.")
    if pd.notna(row.get("spelling_errors")) and row.get("spelling_errors", 0) > 0:
        signals.append(f"Listing has {int(row['spelling_errors'])} recorded spelling error(s).")
    if pd.notna(row.get("seller_rating")) and row.get("seller_rating", 5) < 3.5:
        signals.append(f"Seller rating is {row['seller_rating']:.1f} / 5.")
    if pd.notna(row.get("seller_reviews")) and row.get("seller_reviews", 100) < 50:
        signals.append(f"Seller has {int(row['seller_reviews'])} reviews.")
    return signals, missing


def recommended(row, missing):
    if missing:
        return "Insufficient information"
    return "Investigate" if row["score"] >= 0.5 else "Lower review priority"


def label(value, default="Not available in dataset"):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return default
    return str(value)


def score_text(value):
    if value > 0.9999:
        return ">99.99%"
    if value < 0.0001:
        return "<0.01%"
    return f"{value:.2%}"


st.markdown("""<style>
:root {--ink:#192d46;--muted:#52647b;--line:#d8e1ec;--accent:#234f96;--paper:#fff}
.stApp {background:#f3f6fa;color:var(--ink)}
.stApp, .stApp input, .stApp button, .stApp select {font-family:"Avenir Next","Apple SD Gothic Neo","Malgun Gothic",sans-serif}
.block-container {max-width:1540px;padding:4rem 2.5rem 3rem}
div[data-testid="stVerticalBlock"] {gap:.65rem}
h1,h2,h3 {color:var(--ink);text-wrap:balance;scroll-margin-top:4rem}
h1 {font-size:1.85rem !important;font-weight:650 !important;letter-spacing:-.045em;padding-top:0 !important}
h2,h3 {letter-spacing:-.025em}
h3 {font-size:1.2rem !important}
.eyebrow {color:var(--muted);font-size:.78rem;font-weight:600}
.subtle {color:var(--muted);font-size:.88rem}
.disclaimer {border:1px solid var(--line);background:var(--paper);padding:10px 14px;color:var(--muted);font-size:.82rem;border-radius:8px;margin:.4rem 0}
.reference-placeholder {background:#eaf0f7;border:1px dashed #b1bfd0;border-radius:8px;min-height:220px;display:flex;align-items:center;justify-content:center;color:var(--muted);text-align:center;padding:20px}
.case-summary {display:flex;align-items:center;justify-content:space-between;gap:20px;background:#e9eff8;border:1px solid #c8d7ed;border-radius:8px;padding:16px 20px;margin:.25rem 0}
.recommendation {font-size:1.25rem;font-weight:650;color:var(--ink);line-height:1.4}
.score-detail {text-align:right;flex-shrink:0}
.screening-score {font-size:1.1rem;font-weight:600;font-variant-numeric:tabular-nums}
.ref-label {color:var(--muted);font-size:.76rem;line-height:1.5;margin:.35rem 0}
div.stButton > button, div.stDownloadButton > button {border-radius:7px;min-height:2.65rem;white-space:normal;touch-action:manipulation}
div.stButton > button p {white-space:normal;overflow:visible;text-overflow:clip;overflow-wrap:anywhere}
div.stButton > button[kind="primary"] {background:var(--accent);border-color:var(--accent);color:white}
div.stButton > button[kind="primary"]:hover {background:#193d77;border-color:#193d77}
button:focus-visible,a:focus-visible,input:focus-visible,[role="combobox"]:focus-visible {outline:3px solid #2369c8 !important;outline-offset:3px}
.st-key-case_list {background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:12px}
.st-key-case_list div.stButton > button {justify-content:flex-start;text-align:left}
.st-key-case_list div.stButton > button p {white-space:normal;overflow-wrap:anywhere;font-size:.87rem}
.st-key-case_list [data-testid="stCaptionContainer"] {font-size:.76rem;line-height:1.45}
.st-key-case_reference [data-testid="stImage"] img {max-height:245px;object-fit:contain}
.st-key-decision_panel {background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:18px}
.st-key-queue_toolbar {padding:.6rem 0;border-top:1px solid var(--line)}
[data-testid="stSidebar"] {background:#eaf0f7;border-right:1px solid var(--line)}
[data-testid="stCaptionContainer"] {color:var(--muted)}
hr {border-color:var(--line)}
@media (max-width:760px) {
  .block-container {padding:4rem 1rem 2rem}
  h1 {font-size:1.5rem !important}
  .st-key-case_list {max-height:330px}
  .st-key-catalog_gallery {max-height:520px}
  .case-summary {padding:12px;gap:12px;flex-wrap:wrap}
  .score-detail {text-align:left}
  .st-key-decision_panel {padding:12px}
}
</style>""", unsafe_allow_html=True)
header, language_control = st.columns([5, 1])
with language_control:
    st.selectbox(t("Language"), ["en", "ko"], format_func=lambda value: {"en": "English", "ko": "한국어"}[value], key="language")
if st.session_state.language == "ko":
    st.markdown("""<style>
    [data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p {font-size: 0 !important}
    [data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p::after {content: "파일 선택"; font-size: 14px}
    [data-testid="stFileUploaderDropzoneInstructions"] span {font-size: 0 !important}
    [data-testid="stFileUploaderDropzoneInstructions"] span::after {content: "파일당 최대 200MB · CSV"; font-size: 14px}
    </style>""", unsafe_allow_html=True)
with header:
    st.markdown(f'<div class="eyebrow">{t("Team 9 / Artificial Intelligence and Business")}</div>', unsafe_allow_html=True)
    st.title(t("Brand Protection Intelligence"))
    st.markdown(f'<div class="subtle">{t("Screen & prioritize / Analyst review workspace")}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="disclaimer">{t(DISCLAIMER)}</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f'<div class="eyebrow">{t("Data source")}</div>', unsafe_allow_html=True)
    st.header(t("Dataset"))
    upload = st.file_uploader(t("Upload product CSV"), type="csv", help=t("Use the Kaggle product listing CSV, not the transactions file."), key="dataset_upload")
    st.caption(t("Local CSV only. Data is processed in this app session."))
    st.markdown(f'[{t("Dataset on Kaggle")}](https://www.kaggle.com/datasets/aimlveera/counterfeit-product-detection-dataset)')

paths = [ROOT / "data" / "counterfeit_products.csv", ROOT / "archive" / "counterfeit_products.csv"]
default = next((p for p in paths if p.exists()), None)
data = upload.getvalue() if upload else default.read_bytes() if default else None
source = f"Uploaded CSV: {upload.name}" if upload else f"Local CSV: {default.relative_to(ROOT)}" if default else "No local CSV"
info = None
error = None
if data:
    try:
        cases, model, info = analyze(data, tuple(FEATURES))
    except (ValueError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        error = str(exc)
        model = None
else:
    model = None

if model is None:
    st.warning(t("No usable trained model. Showing clearly synthetic rule-based demonstration cases. Upload a product CSV with a usable binary is_counterfeit target to train the baseline."))
    if error:
        st.caption(t("CSV could not be used: {error}", error=error))
    cases = synthetic_data()
    # This is a simple rule indicator, not an ML probability.
    cases["score"] = cases.apply(lambda row: min(0.9, 0.15 + 0.2 * len(indicators(row)[0])), axis=1)
    source = "Synthetic demonstration records"
    data_key = "synthetic"
    score_name = "Rule-based demonstration score"
else:
    data_key = hashlib.sha256(data).hexdigest()[:12]
    score_name = "Experimental screening score"
    # Keep the original eight IDs attached to their original records for existing sessions.
    fashion = cases[cases.get("category", pd.Series(index=cases.index, dtype=object)).isin(["fashion", "luxury goods"])]
    pool = fashion if len(fashion) >= 8 else cases
    ordered = pool.sort_values("score", ascending=False)
    positions = sorted({round((len(ordered) - 1) * q) for q in [0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95]})
    legacy_ids = {idx: f"T9-{i:03d}" for i, idx in enumerate(ordered.iloc[positions].index, 1)}
    cases = cases.copy()
    cases["case_id"] = [legacy_ids.get(index, f"T9-P-{label(item.get('product_id'), f'row-{index}')}") for index, item in cases.iterrows()]

if model is None:
    cases["case_id"] = [f"T9-{i:03d}" for i in range(1, len(cases) + 1)]
cases = cases.reset_index(drop=True)
if st.session_state.get("data_key") != data_key:
    st.session_state.data_key = data_key
    st.session_state.decisions = {}
    st.session_state.log = []
    initial = cases[cases["category"].astype(str).str.lower() == "fashion"] if "category" in cases else cases
    if initial.empty:
        initial = cases
    st.session_state.selected_case = initial.sort_values("score", ascending=False).iloc[0]["case_id"]

display_source = (t("Synthetic demonstration records") if model is None else
                  t("Uploaded CSV: {name}", name=upload.name) if upload else
                  t("Local CSV: {path}", path=default.relative_to(ROOT)) if default else t("No local CSV"))
st.caption(t("Uploaded listing data · Image references are not evidence" if upload else "Synthetic listing data · Reference images are not evidence"))
catalog_dir = find_catalog(ROOT)
catalog, catalog_stats = build_verified_catalog(str(catalog_dir / "styles.csv"), (catalog_dir / "styles.csv").stat().st_mtime_ns, str(catalog_dir / "images"), (catalog_dir / "images").stat().st_mtime_ns) if catalog_dir else (pd.DataFrame(), {"valid": 0, "examined": 0, "low_resolution": 0, "missing": 0, "invalid": 0, "metadata_incomplete": 0})
if len(catalog) < 100:
    st.warning(t("The high-resolution catalog is unavailable or has fewer than 100 verified products."))

nav1, nav2, workspace_note = st.columns([1.25, 1.25, 2.5], vertical_alignment="center")
if "screen" not in st.session_state:
    st.session_state.screen = "queue"
if nav1.button(t("Investigation queue"), key="nav_queue", type="primary" if st.session_state.screen == "queue" else "secondary", width='stretch'):
    st.session_state.screen = "queue"
    st.rerun()
if nav2.button(t("Product catalog"), key="nav_catalog", type="primary" if st.session_state.screen == "catalog" else "secondary", width='stretch'):
    st.session_state.screen = "catalog"
    st.rerun()
with workspace_note:
    st.caption(t("{count} cases reviewed in this session", count=len(st.session_state.decisions)))

if st.session_state.screen == "catalog":
    st.subheader(t("Separate fashion catalog"))
    st.caption(t("These are catalog products, not the synthetic investigation cases."))
    if catalog.empty:
        st.warning(t("No verified high-resolution images were found. Add the extracted high-resolution dataset under fashion-dataset/."))
        st.stop()
    st.caption(t("Verified catalog: {valid} images; {examined} checked, {low} below 800px on the shorter side, {other} missing or invalid.", valid=catalog_stats['valid'], examined=catalog_stats['examined'], low=catalog_stats['low_resolution'], other=catalog_stats['missing'] + catalog_stats['invalid'] + catalog_stats['metadata_incomplete']))
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        if "catalog_search" not in st.session_state:
            st.session_state.catalog_search = st.session_state.get("catalog_query", "")
        def search_changed():
            st.session_state.catalog_query = st.session_state.catalog_search
            st.session_state.catalog_page = 1
        search = st.text_input(t("Search catalog"), key="catalog_search", on_change=search_changed)
    with f2:
        category = localized_select("catalog_category", "Category", ["All categories"] + sorted(catalog["masterCategory"].unique().tolist()), "catalog_page")
    with f3:
        ordering = localized_select("catalog_order", "Sort catalog", ["Title: A to Z", "Title: Z to A", "Catalog ID"], "catalog_page")
    filtered = catalog
    if category != "All categories":
        filtered = filtered[filtered["masterCategory"] == category]
    if search.strip():
        query = search.strip().casefold()
        searchable = filtered[["id", "productDisplayName", "masterCategory", "subCategory", "articleType", "baseColour", "gender"]].astype(str).agg(" ".join, axis=1).str.casefold()
        filtered = filtered[searchable.str.contains(query, regex=False)]
    filtered = filtered.sort_values("id" if ordering == "Catalog ID" else "productDisplayName", ascending=ordering != "Title: Z to A", kind="stable")
    page = page_for("catalog_page", len(filtered))
    page_items = filtered.iloc[(page - 1) * 24:page * 24]
    st.caption(t("{shown} of {total} products displayed", shown=len(page_items), total=len(filtered)))
    if not len(filtered):
        st.info(t("No products match these filters."))
    gallery, detail = st.columns([2.2, 1], gap="large")
    with gallery, st.container(height=780, border=False, key="catalog_gallery"):
        for offset in range(0, len(page_items), 4):
            columns = st.columns(4, gap="small")
            for col, (_, product) in zip(columns, page_items.iloc[offset:offset + 4].iterrows()):
                with col:
                    thumbnail = image_bytes(product.to_dict(), 280, 280)
                    if thumbnail:
                        st.image(thumbnail, width='stretch', output_format="PNG")
                    else:
                        st.markdown(f'<div class="thumb-placeholder">{t("No image")}</div>', unsafe_allow_html=True)
                    if st.button(product["productDisplayName"], key=f"product_{product['id']}", width='stretch'):
                        st.session_state.catalog_selected = product["id"]
                        st.rerun()
                    st.caption(f"{product['id']} · {product['articleType']}")
    with detail:
        st.markdown(f'<div class="eyebrow">{t("Selected catalog product")}</div>', unsafe_allow_html=True)
        selected_id = st.session_state.get("catalog_selected")
        selected = filtered[filtered["id"] == selected_id] if selected_id else pd.DataFrame()
        if selected.empty and not page_items.empty:
            selected = page_items.iloc[[0]]
            st.session_state.catalog_selected = selected.iloc[0]["id"]
        if selected.empty:
            st.caption(t("Choose a product from the grid."))
        else:
            product = selected.iloc[0]
            image = image_bytes(product.to_dict(), 900, 900)
            if image:
                st.image(image, width='stretch', output_format="PNG")
            else:
                st.markdown(f'<div class="reference-placeholder">{t("No image")}</div>', unsafe_allow_html=True)
            st.subheader(product["productDisplayName"])
            st.caption(f"{t('Image ID')}: {product['id']} · {t('Image dimensions')}: {product['width']} × {product['height']} px")
            st.write(f"{t('Category')}: {product['masterCategory']} / {product['subCategory']}")
            st.write(f"{t('Article type')}: {product['articleType']} · {t('Color')}: {product['baseColour']} · {t('Gender')}: {product['gender']}")
    st.stop()

st.subheader(t("Listing queue"))
st.caption(t("Select a case to inspect the record. Scores are for review prioritization."))
with st.container(key="queue_toolbar"):
    search_column, count_column = st.columns([2, 1], vertical_alignment="bottom")
    with search_column:
        if "case_search" not in st.session_state:
            st.session_state.case_search = st.session_state.get("case_query", "")
        search = st.text_input(t("Search cases"), key="case_search", placeholder=t("Case ID, brand or seller"), on_change=search_cases_changed)
    with count_column:
        st.caption(t("Browseable investigation cases: {count}", count=len(cases)))
    filters = st.columns(4)
    with filters[0]:
        priority_filter = localized_select("priority_filter", "Filter priority", ["All priorities", "Investigate", "Insufficient information", "Lower review priority"], "queue_page")
    with filters[1]:
        if "category_filter" not in st.session_state and "category" in cases and "fashion" in cases["category"].values:
            st.session_state.category_filter = "fashion"
        category_filter = localized_select("category_filter", "Filter category", ["All categories"] + sorted(cases["category"].dropna().astype(str).unique().tolist()) if "category" in cases else ["All categories"], "queue_page")
    with filters[2]:
        status_filter = localized_select("status_filter", "Filter review status", ["All statuses", "Not reviewed", "Investigate", "Information requested", "Dismissed"], "queue_page")
    with filters[3]:
        sort_options = ["Screening score: high to low", "Screening score: low to high", "Review status"]
        language = st.session_state.language
        sort = st.selectbox(t("Sort queue"), sort_options, index=sort_options.index(st.session_state.get("sort_choice", sort_options[0])), format_func=lambda option: translate(language, option), key=f"queue_sort_{language}", on_change=sync_sort)

queue = cases.copy()
queue["status"] = queue["case_id"].map(lambda x: st.session_state.decisions.get(x, "Not reviewed"))
queue["priority"] = queue.apply(lambda item: recommended(item, indicators(item)[1]), axis=1)
if search.strip():
    fields = [field for field in ["case_id", "brand", "seller_id"] if field in queue]
    searchable = queue[fields].fillna("").astype(str).agg(" ".join, axis=1)
    queue = queue[searchable.str.contains(search.strip(), case=False, regex=False)]
if priority_filter != "All priorities":
    queue = queue[queue["priority"] == priority_filter]
if category_filter != "All categories":
    queue = queue[queue["category"].astype(str) == category_filter]
if status_filter != "All statuses":
    queue = queue[queue["status"] == status_filter]
queue = queue.sort_values("status" if sort == "Review status" else "score", ascending=sort != "Screening score: high to low", kind="stable")
pagination, result_count, reset = st.columns([1, 2, 1], vertical_alignment="bottom")
with pagination:
    page = page_for("queue_page", len(queue), 20)
queue_page = queue.iloc[(page - 1) * 20:page * 20]
with result_count:
    st.caption(t("{shown} of {total} cases displayed", shown=len(queue_page), total=len(queue)))
with reset:
    st.button(t("Clear filters"), key="clear_queue_filters", on_click=reset_queue_filters, width="stretch")
if queue.empty:
    st.info(t("No cases match. Change your search or clear the filters."))
selected_row = cases.loc[cases["case_id"] == st.session_state.selected_case].iloc[0]
reference = match_reference(selected_row, catalog, catalog_dir / "images") if catalog_dir and not catalog.empty else None

left, right = st.columns([1.15, 2.85], gap="large")
with left:
    st.markdown(f'<div class="eyebrow">{t("01 / Triage")}</div>', unsafe_allow_html=True)
    st.subheader(t("Case queue"))
    st.caption(t("Choose a case. Review details appear beside the queue."))
    with st.container(height=620, border=False, key="case_list"):
        for _, item in queue_page.iterrows():
            is_selected = item["case_id"] == st.session_state.selected_case
            if st.button(f"{item['case_id']} · {shown(item.get('category')).title()}", key=f"select_{item['case_id']}", width='stretch', type="primary" if is_selected else "secondary"):
                st.session_state.selected_case = item["case_id"]
                st.rerun()
            st.caption(f"{t('Priority')}: {t(item['priority'])} · {t(item['status'])}")
            st.divider()

with right:
    row = cases.loc[cases["case_id"] == st.session_state.selected_case].iloc[0]
    if row["case_id"] not in set(queue_page["case_id"]):
        st.caption(t("The selected case remains available even when it is outside the current page or filters."))
    signals, missing = indicators(row)
    recommendation = recommended(row, missing)
    st.markdown(f'<div class="eyebrow">{t("02 / Case review")}</div>', unsafe_allow_html=True)
    st.subheader(f"{row['case_id']} · {shown(row.get('category')).title()}")
    status = st.session_state.decisions.get(row["case_id"], "Not reviewed")
    st.caption(t("Review status: {status}", status=t(status)))
    display_score = score_text(row["score"]).replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f'<div class="case-summary"><div><div class="subtle">{t("Suggested next step")}</div><div class="recommendation">{t(recommendation)}</div></div><div class="score-detail"><div class="subtle">{t(score_name)}</div><div class="screening-score">{display_score}</div></div></div>', unsafe_allow_html=True)
    st.caption(t("The score prioritizes review. It does not establish infringement."))
    picture, review = st.columns([.85, 1.5], gap="large")
    with picture, st.container(key="case_reference"):
        image = image_bytes(reference, 880, 780)
        if image:
            st.image(image, width='stretch', output_format="PNG")
            st.markdown(f'<div class="ref-label"><strong>{t("Category-matched reference image")}</strong><br>{t("Reference only, not the actual listing or evidence of authenticity.")}</div>', unsafe_allow_html=True)
            st.caption(t("Matched using available listing attributes; the image remains a reference." if reference["specific_match"] else "Broad category match; the actual item type is unknown."))
            st.caption(t("Reference catalog ID: {id}", id=reference["id"]))
        else:
            st.markdown(f'<div class="reference-placeholder">{t("No compatible reference image")}</div>', unsafe_allow_html=True)
            st.caption(t("Reference only, not the actual listing or evidence of authenticity."))
    with review:
        st.caption(f"{t('Brand')}: {shown(row.get('brand'))} · {t('Seller')}: {shown(row.get('seller_id'))}")
        st.markdown(f'**{t("Observable record checks")}**')
        st.markdown("\n".join(f"- {translate_indicator(st.session_state.language, item)}" for item in (signals or ["No listed indicator triggered."])[:3]))
        if len(signals) > 3:
            with st.expander(t("More record checks")):
                for item in signals[3:]:
                    st.write(f"• {translate_indicator(st.session_state.language, item)}")
        st.caption(t("These checks do not explain the model prediction."))
        st.markdown(f'**{t("Human verification required")}**')
        st.markdown("\n".join([*(f"- {translate_indicator(st.session_state.language, item)}" for item in missing), "- " + t("Actual product image, trademark rights, authorized-seller records and listing URL are not available.")]))
        with st.expander(t("Listing details")):
            st.write(f"{t('Brand')}: {shown(row.get('brand'))}")
            st.write(f"{t('Seller')}: {shown(row.get('seller_id'))}")
            st.write(f"{t('Price')}: {shown(row.get('price'))}")
            st.write(f"{t('Seller rating')}: {shown(row.get('seller_rating'))}")
            st.write(f"{t('Seller reviews')}: {shown(row.get('seller_reviews'))}")
    with st.container(key="decision_panel"):
        st.markdown(f'<div class="eyebrow">{t("03 / Human decision")}</div>', unsafe_allow_html=True)
        st.subheader(t("Review decision"))
        st.caption(t("Choose a decision; the recommendation is not applied automatically."))
        buttons = st.columns(3)
        primary = {"Investigate": "Investigate", "Insufficient information": "Information requested", "Lower review priority": "Dismissed"}[recommendation]
        for col, action, status in zip(buttons, ["Mark for investigation", "Request more information", "Dismiss from review"], ["Investigate", "Information requested", "Dismissed"]):
            if col.button(t(action), key=f"action_{status}", width='stretch', type="primary" if status == primary else "secondary"):
                st.session_state.decisions[row["case_id"]] = status
                st.session_state.log.append({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "case_id": row["case_id"], "source_product_id": label(row.get("product_id"), "Unavailable"), "source": source, "score_type": score_name, "score": float(row["score"]), "suggested_next_step": recommendation, "human_decision": status})
                st.rerun()
        if row["case_id"] in st.session_state.decisions:
            st.success(t("Decision recorded: {status}. Choose another action to change it.", status=t(st.session_state.decisions[row["case_id"]])))
        log_columns = ["timestamp_utc", "case_id", "source_product_id", "source", "score_type", "score", "suggested_next_step", "human_decision"]
        log = pd.DataFrame(st.session_state.log, columns=log_columns)
        st.download_button(t("Download review log CSV"), log.to_csv(index=False).encode("utf-8"), "review_log.csv", "text/csv", width='stretch', key="download_log", disabled=log.empty)
        st.caption(t("Decisions stay in this browser session. No complaints, seller contact, or takedowns are initiated."))

with st.expander(t("Model & data details")):
    st.caption(t("{source} · Case IDs and review statuses are synthetic workflow fields. Prices have no verified currency in the dataset.", source=display_source))
    if not upload and default:
        st.caption(t("The Kaggle data card describes these records as synthetic. Listing attributes, brands, sellers and labels are simulated, not marketplace evidence."))
    elif upload:
        st.caption(t("Uploaded CSV provenance is unverified. The named Kaggle dataset is synthetic; do not treat its listings as real marketplace evidence."))
    st.caption(t("This score is not proof of infringement or a legal counterfeit probability. Real-world reliability, generalization, and calibration remain unverified."))
    st.caption(t("Simple record checks, not explanations of the model's prediction. No single indicator establishes counterfeiting."))
    if info:
        raw = info["raw"]
        st.write(t("{rows:,} source rows · {duplicates} exact duplicates removed · {duplicate_ids} rows with repeated product IDs excluded · {excluded_labels} rows with missing or unrecognized labels excluded from training/evaluation.", rows=len(raw), duplicates=info['duplicates'], duplicate_ids=info['duplicate_ids'], excluded_labels=info['excluded_labels']))
        st.write(t("Missing values by column (blank strings included):"))
        st.dataframe(raw.replace(r"^\s*$", np.nan, regex=True).isna().sum().rename(t("missing")).to_frame().join(raw.dtypes.astype(str).rename(t("dtype"))), width='stretch')
        st.write(t("Class distribution after duplicate handling: label 0 = {zero:,}; label 1 = {one:,}.", zero=int(info['counts'].get(0, 0)), one=int(info['counts'].get(1, 0))))
        st.write(t("Allowed features: {features}", features=", ".join(info["features"]) or t("none")))
        st.caption(t("Target, IDs, brand, geography, transaction activity and investigation-like signals are excluded. Seller reviews, image count and description length are also excluded: their ranges perfectly separate labels in the supplied synthetic dataset. Other generated features may still encode label patterns, and repeated-seller dependence cannot be ruled out."))
        if model is not None:
            st.write(t("Held-out evaluation: {train:,} training / {test:,} test rows, stratified split, seed 42. Precision: {precision:.3f}; recall: {recall:.3f}.", train=info['train_size'], test=info['test_size'], precision=info['precision'], recall=info['recall']))
            st.dataframe(pd.DataFrame(info["matrix"], index=[t("Actual 0"), t("Actual 1")], columns=[t("Predicted 0"), t("Predicted 1")]))
    else:
        st.write(t("No CSV was loaded. Download the product CSV from Kaggle, put it in data/counterfeit_products.csv, or upload it above."))
    st.caption(t("The Kaggle data card describes its dataset as synthetic and lists CC BY-SA 4.0. Check the original terms before reuse. Labels are simulated, not legal determinations. Uploaded data provenance is unverified. This is a classroom prototype, not a production system."))

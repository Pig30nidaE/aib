"""Deterministic, category-only references from the separate fashion catalog."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

CATALOG_COLUMNS = ["id", "gender", "masterCategory", "subCategory", "articleType", "baseColour", "productDisplayName"]


def find_catalog(root: Path):
    for directory in (root / "fashion-dataset", root / "fashion-dataset" / "fashion-dataset"):
        if (directory / "styles.csv").is_file() and (directory / "images").is_dir():
            return directory
    return None


@st.cache_data(show_spinner=False)
def load_catalog(csv_path: str, modified_ns: int):
    del modified_ns
    try:
        catalog = pd.read_csv(csv_path, dtype=str, on_bad_lines="skip", usecols=lambda name: name in CATALOG_COLUMNS).fillna("")
    except (OSError, ValueError, UnicodeError, pd.errors.ParserError):
        return pd.DataFrame(columns=CATALOG_COLUMNS)
    if not set(CATALOG_COLUMNS).issubset(catalog.columns):
        return pd.DataFrame(columns=CATALOG_COLUMNS)
    catalog = catalog[catalog["id"].str.fullmatch(r"\d+")].drop_duplicates("id")
    for column in ("masterCategory", "subCategory", "articleType", "baseColour", "gender"):
        catalog[f"_{column}"] = catalog[column].map(normalize)
    return catalog.sort_values("id", kind="stable").reset_index(drop=True)


def normalize(value):
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).casefold().strip().replace("-", " ").split())


@st.cache_data(show_spinner=False)
def build_verified_catalog(csv_path: str, csv_mtime: int, images_dir: str, dir_mtime: int, limit: int = 300, scan_limit: int = 2000):
    """Inspect a deterministic bounded sample, keeping only real high-resolution JPEGs."""
    del dir_mtime
    source = load_catalog(csv_path, csv_mtime)
    stats = {"examined": 0, "valid": 0, "low_resolution": 0, "missing": 0, "invalid": 0, "metadata_incomplete": 0}
    accepted = []
    if source.empty:
        return source, stats
    for _, record in source.sample(frac=1, random_state=42).head(scan_limit).iterrows():
        stats["examined"] += 1
        if not record["productDisplayName"].strip() or not record["masterCategory"].strip():
            stats["metadata_incomplete"] += 1
            continue
        path = Path(images_dir) / f"{record['id']}.jpg"
        if not path.is_file():
            stats["missing"] += 1
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            if min(width, height) < 800:
                stats["low_resolution"] += 1
                continue
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
            stats["invalid"] += 1
            continue
        result = record.to_dict()
        result.update(width=width, height=height, path=str(path))
        accepted.append(result)
        if len(accepted) >= limit:
            break
    stats["valid"] = len(accepted)
    return pd.DataFrame(accepted, columns=[*source.columns, "width", "height", "path"]), stats


def match_reference(row, catalog, images_dir: Path):
    """Match only compatible broad categories; never infer type from a brand or score."""
    if catalog.empty:
        return None
    category = normalize(row.get("category"))
    mappings = {
        "fashion": (("apparel", "footwear", "accessories"), None),
        "clothing": (("apparel",), None),
        "apparel": (("apparel",), None),
        "footwear": (("footwear",), None),
        "shoes": (("footwear",), None),
        "accessories": (("accessories",), None),
        "luxury goods": (("accessories",), ("bags", "watches", "jewellery")),
        "cosmetics": (("personal care",), ("lips", "nails", "makeup", "eyes")),
    }
    if category not in mappings:
        return None
    masters, subcategories = mappings[category]
    candidates = catalog[catalog["_masterCategory"].isin(masters)]
    if subcategories:
        candidates = candidates[candidates["_subCategory"].isin(subcategories)]
    # If the listing specifies a type, require an exact normalized match. Optional
    # color and gender refine compatible candidates only when an exact match exists.
    source_fields = {normalize(k).replace("_", ""): v for k, v in row.items()}
    for source_names, column, required in [
        (("subcategory",), "subCategory", True),
        (("articletype", "producttype"), "articleType", True),
        (("basecolour", "basecolor", "colour", "color"), "baseColour", False),
        (("gender",), "gender", False),
    ]:
        value = next((normalize(source_fields[name]) for name in source_names if name in source_fields and normalize(source_fields[name])), "")
        if value:
            matches = candidates[candidates[f"_{column}"] == value]
            if required or not matches.empty:
                candidates = matches
    if candidates.empty:
        return None
    identity = str(row.get("product_id") or row.get("case_id") or category)
    start = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16) % len(candidates)
    for offset in range(len(candidates)):
        record = candidates.iloc[(start + offset) % len(candidates)]
        image_path = images_dir / f"{record['id']}.jpg"
        if image_path.is_file():
            specific = any(normalize(source_fields.get(name)) for name in ("subcategory", "articletype", "producttype"))
            return {"path": str(image_path), "specific_match": specific, **record.to_dict()}
    return None


@st.cache_data(show_spinner=False, max_entries=128)
def render_reference(path: str, width: int, height: int, modified_ns: int):
    del modified_ns
    try:
        with Image.open(path) as image:
            image.draft("RGB", (width, height))
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (width, height), "#ffffff")
            canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
            result = BytesIO()
            canvas.save(result, format="PNG")
            return result.getvalue()
    except (OSError, ValueError, UnidentifiedImageError):
        return None


def image_bytes(reference, width, height):
    if not reference:
        return None
    try:
        modified_ns = Path(reference["path"]).stat().st_mtime_ns
    except OSError:
        return None
    return render_reference(reference["path"], width, height, modified_ns)

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from image_catalog import build_verified_catalog, find_catalog, image_bytes, load_catalog, match_reference

ROOT = Path(__file__).resolve().parents[1]


class ImageCatalogTests(unittest.TestCase):
    def test_real_catalog_uses_metadata_id_and_compatible_category(self):
        directory = find_catalog(ROOT)
        if directory is None:
            self.skipTest("Fashion image dataset is not installed")
        catalog, stats = build_verified_catalog(str(directory / "styles.csv"), (directory / "styles.csv").stat().st_mtime_ns, str(directory / "images"), (directory / "images").stat().st_mtime_ns)
        self.assertGreaterEqual(len(catalog), 100)
        self.assertEqual(len(catalog), catalog.id.nunique())
        self.assertTrue(((catalog.width >= 800) & (catalog.height >= 800)).all())
        self.assertEqual(stats["valid"], len(catalog))
        source = {"product_id": "demo-123", "category": "Fashion"}
        reference = match_reference(source, catalog, directory / "images")
        self.assertEqual(reference, match_reference(source, catalog, directory / "images"))
        self.assertIn(reference["masterCategory"], ("Apparel", "Footwear", "Accessories"))
        self.assertEqual(Path(reference["path"]).name, f"{reference['id']}.jpg")
        record = catalog.loc[catalog["id"] == reference["id"]].iloc[0]
        self.assertEqual(reference["productDisplayName"], record["productDisplayName"])
        self.assertIsNotNone(image_bytes(reference, 96, 96))
        self.assertIsNone(match_reference({"category": "Electronics"}, catalog, directory / "images"))

    def test_matching_specific_type_and_missing_images(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            folder = Path(temp)
            metadata = pd.DataFrame([
                {"id": "10", "gender": "Men", "masterCategory": "Apparel", "subCategory": "Topwear", "articleType": "Shirts", "baseColour": "Blue", "productDisplayName": "Blue Shirt"},
                {"id": "20", "gender": "Women", "masterCategory": "Footwear", "subCategory": "Shoes", "articleType": "Casual Shoes", "baseColour": "Black", "productDisplayName": "Black Shoes"},
            ])
            metadata.to_csv(folder / "styles.csv", index=False)
            Image.new("RGB", (10, 12), "blue").save(folder / "10.jpg")
            Image.new("RGB", (10, 12), "black").save(folder / "20.jpg")
            catalog = load_catalog(str(folder / "styles.csv"), (folder / "styles.csv").stat().st_mtime_ns)
            verified, stats = build_verified_catalog(str(folder / "styles.csv"), (folder / "styles.csv").stat().st_mtime_ns, str(folder), folder.stat().st_mtime_ns)
            self.assertTrue(verified.empty)
            self.assertEqual(stats["low_resolution"], 2)
            reference = match_reference({"category": "Fashion", "article_type": "Shirts", "color": "Blue", "gender": "Men", "product_id": "x"}, catalog, folder)
            self.assertEqual(reference["id"], "10")
            self.assertIsNone(match_reference({"category": "Fashion", "article_type": "Laptop"}, catalog, folder))
            (folder / "10.jpg").unlink()
            self.assertIsNone(match_reference({"category": "Fashion", "article_type": "Shirts"}, catalog, folder))
            self.assertIsNone(image_bytes(reference, 96, 96))
            (folder / "10.jpg").write_bytes(b"not an image")
            self.assertIsNone(image_bytes(reference, 96, 96))


if __name__ == "__main__":
    unittest.main()

# Brand Protection Intelligence

Team 9's classroom prototype for the **Screen & Prioritize** step of an AI-assisted brand-protection workflow. An analyst can browse an independent fashion catalog or select a synthetic case, review its screening result and record checks, and record a decision. The app does not scrape marketplaces, compare images, contact sellers, file complaints, or initiate takedowns.

## Run locally

Requires Python 3.11 or later. From PowerShell in this project folder:

```powershell
cd D:\AIB_project
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open the local URL printed by Streamlit, normally http://localhost:8501. Stop the server with Ctrl+C. The existing `archive/counterfeit_products.csv` is detected automatically. Alternatively, download the product CSV from [aimlVeera's Ecommerce Counterfeit Products Dataset](https://www.kaggle.com/datasets/aimlveera/counterfeit-product-detection-dataset), extract it to `data/counterfeit_products.csv`, or upload it using the sidebar. The transactions CSV is not used. If no suitable CSV is available, the app uses five explicitly synthetic rule-based examples and does not claim to train a model.

The repository contains the 300 high-resolution images selected for this prototype and their catalog metadata. After updating the app, restart any already-running Streamlit server with Ctrl+C and the command above so it loads the current Python modules.

Use the **Language / 언어** selector at the top right to switch between English and 한국어. Switching languages keeps the selected case and review decisions in the current session. Dataset values and the exported CSV schema and decision codes remain unchanged.

The application text is localized, but Streamlit supplies its own interface controls. Some framework-provided accessibility labels, table toolbar tooltips, and the application menu remain in English in Korean mode. The upload control's visible Korean label is styled for the installed Streamlit version; its screen-reader labels may still be English.

The Kaggle data card describes the dataset itself as **synthetic**. Its listing attributes, brands, sellers and labels do not constitute real marketplace evidence. The local CSV was present in the provided workspace and its filename and structure match the requested dataset; its download provenance has not been independently verified. The data card lists [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Check the source and license terms before redistributing or reusing the data. Uploads have unverified provenance.

## High-resolution catalog and reference images

The app discovers the separate [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) at `fashion-dataset/styles.csv` and `fashion-dataset/images/{id}.jpg`, or under `fashion-dataset/fashion-dataset/`. The repository includes a selected 300-image subset. To use the full dataset, download and extract it separately; the app never downloads it. The local `styles.csv` has 44,424 readable metadata rows. The high-resolution dataset's license was not verified from its data card; check the original terms before reuse.

For responsive startup the app deterministically examines at most 2,000 metadata rows, stopping after finding 300 unique images with a title, matching ID-named JPEG, readable image and at least 800 pixels on the shorter side. In the original dataset it checked 302 records, accepted 300, and excluded two low-resolution images. Only the accepted 300 images are retained in the repository; the removed files are counted as missing when the app scans the subset. The catalog displays 24 products per page, with search, category filter, sorting and product details. Resized previews are generated lazily using Pillow and cached. The app never substitutes the earlier Small dataset or enlarges a low-resolution source. If the high-resolution dataset is missing or has fewer than 100 verified images, the app displays a warning.

The two datasets have unrelated IDs. The app uses the listing category to filter the separate catalog to compatible broad categories, then chooses deterministically using the listing ID and opens the JPEG named by the selected catalog record's exact `id`. If the listing specifies an article type or subcategory, a match is required; color and gender refine the match when available. The provided listings have only broad categories, so the selected image's specific item type, color and gender are **not** known to match the listing. The catalog ID beneath the preview makes the selection traceable. Images are references only, never actual listing evidence or model inputs. Unsupported categories, absent files or unreadable images show a placeholder. Metadata and resized images are cached; image files are opened on demand.

## Model and data quality

The app inspects the CSV and displays its types, missing values, label distribution, duplicate counts, feature list, precision, recall and confusion matrix. The supplied product file contains 5,000 rows, no blank fields, no exact duplicate rows and 3,530 false / 1,470 true simulated labels. Some records explicitly mark seller contact information incomplete even though the CSV field is populated. Repeated product IDs are excluded before splitting; exact duplicates are removed. This prevents obvious duplicated records from becoming independent queue cases, although it cannot identify every duplicate listing.

A logistic regression pipeline uses a fixed, stratified 75/25 train/test split. Numeric imputation/scaling and categorical imputation/one-hot encoding are fitted on training data only. The queue exposes all 1,250 held-out rows from the supplied file, with 20 per page and filters for category, recommendation and review status. It defaults to Fashion; choose All categories to see all cases. The eight previous case IDs remain attached to their original rows, and other IDs derive from source product IDs. IDs and decisions remain stable while navigating the same dataset; changing datasets resets the session's decisions. The permitted model fields are price, seller rating, spelling errors, category, contact completeness and return-policy clarity. The target is used only for training/evaluation, never as an input. IDs, brand, geography, transaction activity and investigation-like fields are excluded. Seller reviews, image count and description length are excluded from training because the supplied data has non-overlapping value ranges across its simulated labels. Other features may also reflect synthetic label-generation rules. The random row split can put the same seller in both partitions.

On the supplied CSV with the installed environment, the held-out precision is 0.989 and recall is 0.967. These are measurements on simulated records, not evidence of real-world effectiveness. They can change with data or dependency versions. Calibration, representativeness, independent seller-level validation and real-world performance are untested. The score is not a legal probability or proof. Visible risk indicators are independent rule checks, not explanations of model coefficients. The 0.5 recommendation threshold is illustrative, not validated. Missing essential fields or incomplete contact information produce `Insufficient information` regardless of the score. A low rating or any other single indicator does not establish infringement.

The listing dataset includes image counts but no actual listing photos, verified trademark rights, authorized-seller records or listing URLs. The separate catalog images are only category references. Price currency is not verified. Queue case IDs and review statuses are generated by the app. Decisions are held in Streamlit session state and disappear when the session resets; export the review log before closing. The CSV log includes the source product ID when available, score and every decision event.

## Suggested 3–4 minute demonstration

1. Select a Fashion listing in **Investigation queue**. Cases are paginated, and the category filter can be changed to see other categories.
2. Explain that the score is produced by the model on a held-out synthetic record. Read the spelling and rating indicators, which are simple checks rather than model explanations.
3. Point out incomplete seller contact and the unavailable images and rights records. The app recommends `Insufficient information` despite the high score.
4. Click **Request more information** and show the queue and case status update. Download the review log, and show the recorded decision.
5. Open **Model & data details** to discuss the synthetic data, excluded leakage-prone features, class balance and limits of the held-out metrics.

Switch to **Product catalog** to search and browse the separate high-resolution images. Catalog products are not linked to investigation cases.

To run the localization and image-matching checks, use `python -m unittest discover -s tests -v` from the project folder.

## Peer feedback template

Have another group use the app, then record their own observations:

- Could you understand which case to inspect and why? What was unclear?
- Did the score, indicators and missing evidence help you choose a next action? Why or why not?
- Did the interface make the synthetic data and uncertainty clear?
- What would you change first? What additional evidence would you need?


import csv
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
import pytesseract

st.set_page_config(
    page_title="Reel Comment Collector",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Instagram Reel Comment Collector")
st.write(
    "Collect visible comments from screenshots, copied text, or CSV files. "
    "Review the records and export an evidence log."
)

st.warning(
    "This app does not log in to Instagram or bypass access controls. "
    "For another person's Reel, use comments you can view and are authorized "
    "to document. OCR results must be checked against the originals."
)

COLUMNS = [
    "record_id",
    "username",
    "comment_text",
    "timestamp",
    "reel_url",
    "source_type",
    "source_filename",
    "ocr_text",
    "verified",
    "notes",
    "collected_at_utc",
]


def clean(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def make_id(row):
    fields = [
        "username",
        "comment_text",
        "timestamp",
        "reel_url",
        "source_filename",
        "source_type",
    ]
    raw = "|".join(clean(row.get(k, "")) for k in fields)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def normalize_frame(frame):
    frame = frame.copy()

    # Accept common column names from imported CSV files.
    aliases = {
        "comment": "comment_text",
        "text": "comment_text",
        "user": "username",
        "created_at": "timestamp",
        "url": "reel_url",
        "source_file": "source_filename",
    }

    for old, new in aliases.items():
        if old in frame.columns and new not in frame.columns:
            frame = frame.rename(columns={old: new})

    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    frame = frame[COLUMNS].copy()

    for column in COLUMNS:
        if column != "verified":
            frame[column] = frame[column].map(clean)

    frame["verified"] = frame["verified"].map(
        lambda value: (
            value
            if isinstance(value, bool)
            else str(value).strip().lower()
            in {"true", "1", "yes", "y"}
        )
    )

    for index in frame.index:
        if not frame.at[index, "record_id"]:
            frame.at[index, "record_id"] = make_id(
                frame.loc[index].to_dict()
            )

    return frame


def new_record(**kwargs):
    record = {column: "" for column in COLUMNS}
    record.update(kwargs)
    record["collected_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    record["verified"] = False
    record["record_id"] = make_id(record)
    return record


if "records" not in st.session_state:
    st.session_state.records = pd.DataFrame(columns=COLUMNS)

if "original_screenshots" not in st.session_state:
    st.session_state.original_screenshots = {}

with st.sidebar:
    st.header("Collection details")

    reel_url = st.text_input(
        "Reel URL",
        placeholder="https://www.instagram.com/reel/...",
    )

    case_label = st.text_input(
        "Collection label",
        placeholder="Example: Reel comments, October",
    )

    st.caption(
        "Use a consistent label so related records are easy to identify."
    )

screenshots_tab, paste_tab, csv_tab, review_tab = st.tabs(
    [
        "1. Screenshot OCR",
        "2. Live-view import",
        "3. CSV import",
        "4. Review and export",
    ]
)

# ---------------------------------------------------------
# 1. Screenshot OCR
# ---------------------------------------------------------
with screenshots_tab:
    st.subheader("Upload screenshots in batches")

    uploaded_images = st.file_uploader(
        "Select comment screenshots",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="screenshot_upload",
    )

    st.caption(
        "OCR extracts text visible in an image. It does not reliably "
        "separate usernames, timestamps, and comments automatically."
    )

    if uploaded_images:
        if st.button(
            "Extract text from screenshots",
            type="primary",
            key="run_ocr",
        ):
            try:
                with st.spinner("Reading screenshots..."):
                    new_rows = []

                    for uploaded in uploaded_images:
                        image_bytes = uploaded.getvalue()

                        # Keep the original image bytes for the evidence ZIP.
                        st.session_state.original_screenshots[
                            uploaded.name
                        ] = image_bytes

                        image = Image.open(
                            io.BytesIO(image_bytes)
                        ).convert("RGB")

                        extracted_text = pytesseract.image_to_string(
                            image,
                            config="--psm 6",
                        ).strip()

                        new_rows.append(
                            new_record(
                                username="",
                                comment_text="",
                                timestamp="",
                                reel_url=reel_url,
                                source_type="screenshot_ocr",
                                source_filename=uploaded.name,
                                ocr_text=extracted_text,
                                notes=(
                                    "OCR transcription. Separate and verify "
                                    "individual comments against the original."
                                ),
                            )
                        )

                if new_rows:
                    current = st.session_state.records
                    combined = pd.concat(
                        [current, pd.DataFrame(new_rows)],
                        ignore_index=True,
                    )
                    st.session_state.records = normalize_frame(combined)

                    st.success(
                        f"Processed {len(new_rows)} screenshot(s). "
                        "Open Review and export to verify the text."
                    )

            except Exception as error:
                st.error(
                    "OCR could not run. Check that Tesseract OCR is "
                    f"installed and available. Details: {error}"
                )

        st.write("Selected files:")
        for uploaded in uploaded_images:
            st.write(f"- {uploaded.name}")

# ---------------------------------------------------------
# 2. Live-view import: paste comments visible in Instagram
# ---------------------------------------------------------
with paste_tab:
    st.subheader("Import comments while viewing the Reel")

    st.write(
        "Open the Reel in Instagram and copy comments you can see. "
        "Paste them below, preferably one comment per line."
    )

    st.caption(
        "Supported examples: @username: comment text, or plain comment "
        "text. Check the username and timestamp fields afterward."
    )

    pasted_text = st.text_area(
        "Paste visible comments",
        height=240,
        placeholder=(
            "@account1: Example comment one\n"
            "@account2: Example comment two"
        ),
        key="pasted_comments",
    )

    paste_reel_url = st.text_input(
        "Reel URL for pasted comments",
        value=reel_url,
        key="paste_reel_url",
    )

    if st.button(
        "Add pasted comments",
        type="primary",
        key="add_pasted",
    ):
        lines = [
            line.strip()
            for line in pasted_text.splitlines()
            if line.strip()
        ]

        new_rows = []

        for line in lines:
            username = ""
            comment_text = line

            match = re.match(
                r"^@?([A-Za-z0-9._]{1,30})\s*:\s*(.+)$",
                line,
            )

            if match:
                username = match.group(1)
                comment_text = match.group(2).strip()

            new_rows.append(
                new_record(
                    username=username,
                    comment_text=comment_text,
                    reel_url=paste_reel_url,
                    source_type="copied_text",
                    notes=(
                        "Copied from visible comments. "
                        "Verify against the original Reel."
                    ),
                )
            )

        if new_rows:
            combined = pd.concat(
                [
                    st.session_state.records,
                    pd.DataFrame(new_rows),
                ],
                ignore_index=True,
            )

            st.session_state.records = normalize_frame(combined)

            st.success(
                f"Added {len(new_rows)} comment record(s)."
            )

    st.divider()

    st.subheader("About direct collection")

    st.write(
        "Instagram's official API has access and permission restrictions. "
        "It does not grant unrestricted access to all comments on any "
        "arbitrary public Reel. This app therefore does not pretend to "
        "retrieve comments directly from Instagram. The import workflow "
        "works with comments you can actually view."
    )

# ---------------------------------------------------------
# 3. CSV import
# ---------------------------------------------------------
with csv_tab:
    st.subheader("Import a comment export")

    st.write(
        "Import a CSV from a legitimate export or another tool. "
        "Accepted column names include username/user, "
        "comment_text/comment/text, timestamp/created_at, and "
        "reel_url/url."
    )

    csv_upload = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="csv_upload",
    )

    if csv_upload:
        try:
            imported_df = pd.read_csv(
                csv_upload,
                dtype=str,
                keep_default_na=False,
            )

            st.write("Preview:")
            st.dataframe(
                imported_df.head(10),
                use_container_width=True,
                hide_index=True,
            )

            if st.button(
                "Add CSV records",
                type="primary",
                key="add_csv",
            ):
                imported_df = imported_df.rename(
                    columns={
                        "comment": "comment_text",
                        "text": "comment_text",
                        "user": "username",
                        "created_at": "timestamp",
                        "url": "reel_url",
                    }
                )

                imported_df["source_type"] = "csv_import"

                if "reel_url" not in imported_df.columns:
                    imported_df["reel_url"] = reel_url
                else:
                    imported_df["reel_url"] = (
                        imported_df["reel_url"]
                        .replace("", reel_url)
                    )

                imported_df["collected_at_utc"] = (
                    datetime.now(timezone.utc).isoformat()
                )

                imported_df = normalize_frame(imported_df)

                combined = pd.concat(
                    [
                        st.session_state.records,
                        imported_df,
                    ],
                    ignore_index=True,
                )

                st.session_state.records = normalize_frame(combined)

                st.success(
                    f"Imported {len(imported_df)} row(s)."
                )

        except Exception as error:
            st.error(f"Could not import CSV: {error}")

# ---------------------------------------------------------
# 4. Review, search, and export
# ---------------------------------------------------------
with review_tab:
    st.subheader("Review and manage evidence")

    df = normalize_frame(st.session_state.records)

    if df.empty:
        st.info(
            "No records yet. Upload screenshots, paste visible comments, "
            "or import a CSV using the other tabs."
        )

    else:
        col1, col2, col3 = st.columns(3)

        col1.metric("Total records", len(df))
        col2.metric("Verified records", int(df["verified"].sum()))
        col3.metric(
            "Awaiting verification",
            int((~df["verified"]).sum()),
        )

        search_term = st.text_input(
            "Search comments, usernames, OCR text, filenames, or notes"
        )

        if search_term.strip():
            searchable = df.astype(str).apply(
                lambda column: column.str.contains(
                    re.escape(search_term),
                    case=False,
                    na=False,
                )
            ).any(axis=1)

            st.write(f"Matching records: {int(searchable.sum())}")

            st.dataframe(
                df.loc[searchable],
                use_container_width=True,
                hide_index=True,
            )

        st.write(
            "Edit the records below. For OCR rows, split the OCR text "
            "into individual comment records when necessary. Mark a row "
            "verified only after checking the original."
        )

        edited_df = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="evidence_editor",
            column_config={
                "verified": st.column_config.CheckboxColumn(
                    "Verified",
                    help="Checked against the original source",
                ),
                "comment_text": st.column_config.TextColumn(
                    "Comment",
                    width="large",
                ),
                "ocr_text": st.column_config.TextColumn(
                    "Raw OCR text",
                    width="large",
                ),
                "notes": st.column_config.TextColumn(
                    "Notes",
                    width="large",
                ),
            },
        )

        st.session_state.records = normalize_frame(edited_df)
        output_df = normalize_frame(st.session_state.records)

        csv_bytes = output_df.to_csv(
            index=False,
            quoting=csv.QUOTE_MINIMAL,
        ).encode("utf-8-sig")

        json_bytes = output_df.to_json(
            orient="records",
            force_ascii=False,
            indent=2,
        ).encode("utf-8")

        st.download_button(
            "Download evidence log (CSV)",
            data=csv_bytes,
            file_name="instagram_comment_evidence.csv",
            mime="text/csv",
            type="primary",
        )

        st.download_button(
            "Download evidence log (JSON)",
            data=json_bytes,
            file_name="instagram_comment_evidence.json",
            mime="application/json",
        )

        st.divider()
        st.subheader("Evidence package")

        st.write(
            "Create a ZIP containing the CSV, JSON, collection metadata, "
            "and original screenshots uploaded during this session."
        )

        if st.button("Prepare evidence ZIP"):
            zip_buffer = io.BytesIO()

            metadata = {
                "collection_label": case_label,
                "reel_url": reel_url,
                "created_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "record_count": len(output_df),
                "notice": (
                    "OCR and copied text are unverified until checked "
                    "against original sources."
                ),
            }

            with zipfile.ZipFile(
                zip_buffer,
                "w",
                zipfile.ZIP_DEFLATED,
            ) as archive:
                archive.writestr(
                    "instagram_comment_evidence.csv",
                    csv_bytes,
                )

                archive.writestr(
                    "instagram_comment_evidence.json",
                    json_bytes,
                )

                archive.writestr(
                    "collection_metadata.json",
                    json.dumps(metadata, indent=2),
                )

                for filename, image_bytes in (
                    st.session_state.original_screenshots.items()
                ):
                    safe_filename = Path(filename).name

                    archive.writestr(
                        f"original_screenshots/{safe_filename}",
                        image_bytes,
                    )

            st.download_button(
                "Download evidence ZIP",
                data=zip_buffer.getvalue(),
                file_name="instagram_comment_evidence_package.zip",
                mime="application/zip",
                type="primary",
            )

st.divider()

st.caption(
    "Evidence best practice: keep original screenshots, capture relevant "
    "context, and preserve the source URL and timestamps when available. "
    "An OCR transcript or record ID does not independently establish "
    "authenticity or authorship. Download your work before ending the "
    "session; Streamlit session data is not permanent storage."
)

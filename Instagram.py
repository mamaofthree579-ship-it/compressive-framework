
import csv
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image
from pytesseract import Output


# =========================================================
# APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Instagram Reel Comment Collector",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Instagram Reel Comment Collector")
st.write(
    "Extract multiple visible comments from screenshots, "
    "import copied comments, review records, and export evidence."
)

st.info(
    "This app processes screenshots and text you provide. "
    "It does not log in to Instagram, bypass access controls, "
    "or automatically retrieve comments from another person's Reel."
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

INTERFACE_TEXT = {
    "reply",
    "replies",
    "like",
    "likes",
    "liked by",
    "see translation",
    "view replies",
    "hide replies",
    "follow",
    "following",
    "message",
    "share",
    "send",
    "more",
    "less",
    "report",
}


# =========================================================
# RECORD HELPERS
# =========================================================

def clean(value):
    """Normalize whitespace without converting missing values to text."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return re.sub(r"\s+", " ", str(value)).strip()


def make_id(record):
    """Create a repeatable identifier for a collected record."""
    fields = [
        "username",
        "comment_text",
        "timestamp",
        "reel_url",
        "source_filename",
        "source_type",
    ]

    raw = "|".join(clean(record.get(k, "")) for k in fields)

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]


def new_record(**kwargs):
    """Create a record with all expected fields."""
    record = {column: "" for column in COLUMNS}
    record.update(kwargs)

    record["verified"] = False
    record["collected_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    record["record_id"] = make_id(record)

    return record


def normalize_frame(frame):
    """Ensure all imported and edited records use a common schema."""
    frame = frame.copy()

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


# =========================================================
# OCR HELPERS
# =========================================================

def is_interface_line(text):
    """Identify common interface labels that are not comment text."""
    text = clean(text).lower().strip()

    if not text:
        return True

    if text in INTERFACE_TEXT:
        return True

    if re.match(
        r"^(view all|view more|hide replies|see all|load more)\b",
        text,
    ):
        return True

    if re.match(
        r"^\d+\s+(likes?|repl(?:y|ies))\b",
        text,
    ):
        return True

    if re.match(
        r"^\d+\s*(s|m|h|d|w|mo|y)$",
        text,
    ):
        return True

    return False


def extract_ocr_lines(image):
    """
    Extract OCR text while retaining approximate vertical and
    horizontal positions. Each result represents a detected line.
    """
    data = pytesseract.image_to_data(
        image,
        output_type=Output.DATAFRAME,
        config="--psm 11",
    )

    data = data.dropna(subset=["text"]).copy()
    data["text"] = data["text"].astype(str).str.strip()

    data = data[data["text"] != ""]

    if data.empty:
        return []

    data["conf"] = pd.to_numeric(
        data["conf"],
        errors="coerce",
    ).fillna(-1)

    data = data[data["conf"] >= 15].copy()

    if data.empty:
        return []

    # Tesseract line coordinates group words into reading lines.
    group_columns = [
        "block_num",
        "par_num",
        "line_num",
    ]

    lines = []

    for _, group in data.groupby(
        group_columns,
        sort=False,
    ):
        group = group.sort_values("left")

        text = " ".join(
            group["text"].tolist()
        ).strip()

        if not text:
            continue

        lines.append({
            "text": text,
            "left": int(group["left"].min()),
            "top": int(group["top"].min()),
            "right": int(
                (
                    group["left"] + group["width"]
                ).max()
            ),
            "bottom": int(
                (
                    group["top"] + group["height"]
                ).max()
            ),
        })

    lines.sort(
        key=lambda line: (
            line["top"],
            line["left"],
        )
    )

    return lines


def extract_comments_from_image(image, filename, reel_url):
    """
    Find visible @username anchors and make a separate record
    for each detected account.

    The text after an anchor and the following lines are assigned
    to that account until the next username anchor.

    All results remain unverified for manual review.
    """
    lines = extract_ocr_lines(image)

    if not lines:
        return []

    # Match Instagram-style handles, e.g. @account.name_123.
    handle_pattern = re.compile(
        r"(?<![A-Za-z0-9._])@([A-Za-z0-9._]{1,30})"
    )

    anchors = []

    for line_index, line in enumerate(lines):
        for match in handle_pattern.finditer(line["text"]):
            anchors.append({
                "line_index": line_index,
                "username": match.group(1),
                "match_start": match.start(),
                "match_end": match.end(),
                "top": line["top"],
                "left": line["left"],
            })

    if not anchors:
        return []

    # Sort anchors in approximate reading order.
    anchors.sort(
        key=lambda anchor: (
            anchor["top"],
            anchor["left"],
            anchor["match_start"],
        )
    )

    records = []

    for anchor_number, anchor in enumerate(anchors):
        current_line_index = anchor["line_index"]
        current_line = lines[current_line_index]

        # The next anchor is the boundary for the current record.
        next_anchor = (
            anchors[anchor_number + 1]
            if anchor_number + 1 < len(anchors)
            else None
        )

        if next_anchor is None:
            next_line_index = len(lines)
        else:
            next_line_index = next_anchor["line_index"]

        comment_parts = []

        # Handle text that follows the username on the same line.
        # If another username is on this line, stop before it.
        line_text = current_line["text"]
        start = anchor["match_end"]
        end = len(line_text)

        same_line_next = (
            next_anchor is not None
            and next_anchor["line_index"] == current_line_index
        )

        if same_line_next:
            end = next_anchor["match_start"]

        remainder = line_text[start:end].strip(
            " :|-–—\t"
        )

        if remainder and not is_interface_line(remainder):
            comment_parts.append(remainder)

        # Collect following lines until the next detected username.
        # Do not consume a second anchor on the same line.
        for line_index in range(
            current_line_index + 1,
            next_line_index,
        ):
            text = lines[line_index]["text"].strip()

            if not text:
                continue

            if is_interface_line(text):
                continue

            # If OCR placed a username in the line without correctly
            # registering its anchor, avoid swallowing that line.
            if handle_pattern.search(text):
                continue

            comment_parts.append(text)

        comment_text = clean(" ".join(comment_parts))

        notes = (
            "Detected @username anchor. Verify the account, "
            "comment boundaries, and text against the screenshot."
        )

        if not comment_text:
            notes += " No comment text confidently detected."

        record = new_record(
            username=anchor["username"],
            comment_text=comment_text,
            timestamp="",
            reel_url=reel_url,
            source_type="screenshot_ocr",
            source_filename=filename,
            ocr_text=comment_text,
            verified=False,
            notes=notes,
        )

        records.append(record)

    return records


# =========================================================
# SESSION STATE
# =========================================================

if "records" not in st.session_state:
    st.session_state.records = pd.DataFrame(
        columns=COLUMNS
    )

if "original_screenshots" not in st.session_state:
    st.session_state.original_screenshots = {}


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("Collection details")

    reel_url = st.text_input(
        "Instagram Reel URL",
        placeholder="https://www.instagram.com/reel/...",
    )

    case_label = st.text_input(
        "Collection / case label",
        placeholder="Reel comments — October 2026",
    )

    st.caption(
        "Download your records regularly. Session data is not "
        "permanent storage."
    )


# =========================================================
# TABS
# =========================================================

screenshots_tab, paste_tab, csv_tab, review_tab = st.tabs([
    "1 · Screenshot OCR",
    "2 · Live-view import",
    "3 · CSV import",
    "4 · Review & export",
])


# =========================================================
# TAB 1: SCREENSHOT OCR
# =========================================================

with screenshots_tab:
    st.subheader("Extract multiple comments from screenshots")

    uploaded_images = st.file_uploader(
        "Upload one or more screenshots",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp",
            "bmp",
            "tif",
            "tiff",
        ],
        accept_multiple_files=True,
        key="screenshot_upload",
    )

    st.write(
        "The OCR looks for visible @username anchors and creates "
        "a separate record for each detected anchor."
    )

    st.caption(
        "OCR can still miss handles or misassign text when comments "
        "wrap across lines, usernames are absent, or replies are nested."
    )

    if uploaded_images:
        if st.button(
            "Extract comments from all screenshots",
            type="primary",
            key="run_ocr",
        ):
            new_rows = []
            progress = st.progress(0)
            status = st.empty()

            for index, uploaded in enumerate(uploaded_images):
                try:
                    image_bytes = uploaded.getvalue()

                    # Preserve the exact uploaded image bytes.
                    st.session_state.original_screenshots[
                        uploaded.name
                    ] = image_bytes

                    image = Image.open(
                        io.BytesIO(image_bytes)
                    ).convert("RGB")

                    found_records = extract_comments_from_image(
                        image=image,
                        filename=uploaded.name,
                        reel_url=reel_url,
                    )

                    if found_records:
                        new_rows.extend(found_records)
                    else:
                        # Keep an OCR-only record so the image isn't
                        # silently lost when no @handle is detected.
                        lines = extract_ocr_lines(image)

                        raw_text = "\n".join(
                            line["text"] for line in lines
                        )

                        new_rows.append(
                            new_record(
                                username="",
                                comment_text="",
                                reel_url=reel_url,
                                source_type="screenshot_ocr_review",
                                source_filename=uploaded.name,
                                ocr_text=raw_text,
                                notes=(
                                    "No @username anchor detected. "
                                    "Review this screenshot manually."
                                ),
                            )
                        )

                except Exception as error:
                    st.error(
                        f"Could not process {uploaded.name}: {error}"
                    )

                progress.progress(
                    (index + 1) / len(uploaded_images)
                )

                status.write(
                    f"Processed {index + 1} of "
                    f"{len(uploaded_images)} screenshots"
                )

            if new_rows:
                combined = pd.concat(
                    [
                        st.session_state.records,
                        pd.DataFrame(new_rows),
                    ],
                    ignore_index=True,
                )

                st.session_state.records = normalize_frame(
                    combined
                )

                st.success(
                    f"Created {len(new_rows)} record(s) from "
                    f"{len(uploaded_images)} screenshot(s)."
                )

                st.write(
                    "Open the Review & export tab to correct "
                    "the usernames and comment boundaries."
                )

    st.divider()

    st.subheader("How screenshot extraction works")

    st.markdown(
        """
        - Finds text matching visible `@username` patterns.
        - Uses OCR line coordinates to preserve reading order.
        - Creates a record for each detected username.
        - Associates subsequent lines with that record until the next
          detected username.
        - Retains the original screenshot filename for review.
        """
    )


# =========================================================
# TAB 2: LIVE-VIEW / COPIED TEXT IMPORT
# =========================================================

with paste_tab:
    st.subheader("Import comments visible in Instagram")

    st.write(
        "While viewing the Reel, copy comments you can see and paste "
        "them below. The app turns each non-empty line into a record."
    )

    pasted_text = st.text_area(
        "Paste comments — one comment per line",
        height=240,
        placeholder=(
            "@account_one: First comment text\n"
            "@account_two: Second comment text\n"
            "@account_three: Third comment text"
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
                        "Imported from copied visible text. "
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

            st.session_state.records = normalize_frame(
                combined
            )

            st.success(
                f"Added {len(new_rows)} comment record(s)."
            )

    st.divider()

    st.subheader("About direct Instagram retrieval")

    st.write(
        "This workflow imports text you can view and copy. It is not "
        "a live Instagram connection. Instagram's official API does "
        "not grant unrestricted access to comments on arbitrary "
        "accounts' Reels."
    )


# =========================================================
# TAB 3: CSV IMPORT
# =========================================================

with csv_tab:
    st.subheader("Import an existing comment CSV")

    csv_upload = st.file_uploader(
        "Select a CSV file",
        type=["csv"],
        key="csv_upload",
    )

    if csv_upload is not None:
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
                "Add imported CSV records",
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

                st.session_state.records = normalize_frame(
                    combined
                )

                st.success(
                    f"Imported {len(imported_df)} record(s)."
                )

        except Exception as error:
            st.error(
                f"Could not import CSV: {error}"
            )


# =========================================================
# TAB 4: REVIEW AND EXPORT
# =========================================================

with review_tab:
    st.subheader("Review, search, verify, and export")

    df = normalize_frame(
        st.session_state.records
    )

    if df.empty:
        st.info(
            "No records yet. Use Screenshot OCR, Live-view import, "
            "or CSV import to add comments."
        )

    else:
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total records",
            len(df),
        )

        col2.metric(
            "Verified records",
            int(df["verified"].sum()),
        )

        col3.metric(
            "Unverified records",
            int((~df["verified"]).sum()),
        )

        search_term = st.text_input(
            "Search usernames, comments, OCR text, filenames, or notes"
        )

        if search_term.strip():
            matches = df.astype(str).apply(
                lambda column: column.str.contains(
                    re.escape(search_term),
                    case=False,
                    na=False,
                )
            ).any(axis=1)

            st.write(
                f"Matching records: {int(matches.sum())}"
            )

            st.dataframe(
                df.loc[matches],
                use_container_width=True,
                hide_index=True,
            )

        st.write(
            "Edit the table to correct usernames and comment text. "
            "For OCR-only records, create separate rows for each "
            "comment as needed. Mark Verified only after checking "
            "the original source."
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
                    help="Checked against the original screenshot or source",
                ),
                "comment_text": st.column_config.TextColumn(
                    "Comment text",
                    width="large",
                ),
                "ocr_text": st.column_config.TextColumn(
                    "OCR text",
                    width="large",
                ),
                "notes": st.column_config.TextColumn(
                    "Notes",
                    width="large",
                ),
            },
        )

        st.session_state.records = normalize_frame(
            edited_df
        )

        output_df = normalize_frame(
            st.session_state.records
        )

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

        st.subheader("Evidence ZIP package")

        st.write(
            "The ZIP contains the CSV, JSON, metadata, and original "
            "screenshots uploaded during this app session."
        )

        if st.button(
            "Prepare evidence ZIP",
            key="prepare_zip",
        ):
            zip_buffer = io.BytesIO()

            metadata = {
                "collection_label": case_label,
                "reel_url": reel_url,
                "created_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "record_count": len(output_df),
                "notice": (
                    "OCR and copied text require verification against "
                    "the original source."
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
                    json.dumps(
                        metadata,
                        indent=2,
                    ),
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


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Preserve original screenshots and source context. OCR output "
    "is not proof of authenticity or authorship. Check records "
    "against the original source. Download records before ending "
    "the session because session state is temporary."
)

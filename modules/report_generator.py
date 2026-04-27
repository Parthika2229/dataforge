"""
InsOps · Compliance Report Generator
Generates a professional PDF audit/compliance report using ReportLab.
"""
import io
from datetime import datetime
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ── Brand colors ──────────────────────────────────────────────────────────────
PURPLE   = colors.HexColor("#6c63ff")
DARK     = colors.HexColor("#1a1a2e")
ACCENT   = colors.HexColor("#ff6584")
LIGHT_BG = colors.HexColor("#f7f7fa")
GREEN    = colors.HexColor("#27ae60")
ORANGE   = colors.HexColor("#e67e22")
RED      = colors.HexColor("#e74c3c")
GREY     = colors.HexColor("#7f8c8d")


def _risk_color(level: str):
    return {"High": RED, "Medium": ORANGE, "Low": GREEN}.get(level, GREY)


def _score_color(score: int):
    if score >= 80: return GREEN
    if score >= 50: return ORANGE
    return RED


def generate_report(
    dataset_name:    str,
    raw_shape:       tuple,
    clean_shape:     tuple,
    cleaning_report: list,
    quality_score:   int,
    insights:        list,
    pii_df:          pd.DataFrame,
    compliance_scores: dict,
    ml_results:      dict | None = None,
) -> bytes:
    """
    Build and return a PDF compliance report as bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Helper styles ─────────────────────────────────────────────────────────
    h1 = ParagraphStyle("H1", fontSize=22, textColor=DARK, fontName="Helvetica-Bold",
                         spaceAfter=4, leading=26)
    h2 = ParagraphStyle("H2", fontSize=13, textColor=PURPLE, fontName="Helvetica-Bold",
                         spaceAfter=4, spaceBefore=14, leading=18)
    h3 = ParagraphStyle("H3", fontSize=10, textColor=DARK, fontName="Helvetica-Bold",
                         spaceAfter=3, spaceBefore=6)
    body = ParagraphStyle("Body", fontSize=9, textColor=DARK, fontName="Helvetica",
                           spaceAfter=3, leading=14)
    small = ParagraphStyle("Small", fontSize=8, textColor=GREY, fontName="Helvetica",
                            spaceAfter=2, leading=12)
    center = ParagraphStyle("Center", fontSize=9, alignment=TA_CENTER,
                             fontName="Helvetica", textColor=DARK)

    def hr():
        return HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0f0"),
                          spaceAfter=8, spaceBefore=8)

    def section(title):
        return [Spacer(1, 0.3*cm), Paragraph(title, h2), hr()]

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("⚡ InsOps", ParagraphStyle("Brand", fontSize=28,
                  textColor=PURPLE, fontName="Helvetica-Bold", alignment=TA_CENTER)))
    story.append(Paragraph("AI-Powered Data Pipeline Platform",
                  ParagraphStyle("Sub", fontSize=11, textColor=GREY,
                  fontName="Helvetica", alignment=TA_CENTER, spaceAfter=6)))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=PURPLE))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("DATA PIPELINE & COMPLIANCE REPORT",
                  ParagraphStyle("Title", fontSize=16, textColor=DARK,
                  fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=4)))

    meta = [
        ["Dataset",   dataset_name],
        ["Generated", datetime.now().strftime("%B %d, %Y  %H:%M UTC")],
        ["Raw Shape", f"{raw_shape[0]:,} rows × {raw_shape[1]} columns"],
        ["Clean Shape", f"{clean_shape[0]:,} rows × {clean_shape[1]} columns"],
    ]
    meta_tbl = Table(meta, colWidths=[4*cm, 12*cm])
    meta_tbl.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",  (0,0), (-1,-1), 9),
        ("TEXTCOLOR", (0,0), (0,-1), GREY),
        ("TEXTCOLOR", (1,0), (1,-1), DARK),
        ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [LIGHT_BG, colors.white]),
        ("GRID",      (0,0), (-1,-1), 0.3, colors.HexColor("#ddd")),
        ("TOPPADDING",(0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]))
    story.append(Spacer(1, 0.4*cm))
    story.append(meta_tbl)

    # ── 1. Data Quality Score ─────────────────────────────────────────────────
    story += section("1. Data Quality Score")
    q_color = _score_color(quality_score)
    score_data = [["Quality Score", f"{quality_score} / 100",
                   "GOOD" if quality_score >= 80 else ("FAIR" if quality_score >= 50 else "POOR")]]
    score_tbl = Table(score_data, colWidths=[5*cm, 5*cm, 6*cm])
    score_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), LIGHT_BG),
        ("FONTNAME",     (0,0), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 14),
        ("TEXTCOLOR",    (1,0), (1,0),   q_color),
        ("TEXTCOLOR",    (2,0), (2,0),   q_color),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 12),
        ("BOTTOMPADDING",(0,0), (-1,-1), 12),
        ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#ddd")),
    ]))
    story.append(score_tbl)

    # ── 2. Cleaning Summary ───────────────────────────────────────────────────
    story += section("2. Automated Cleaning Summary")
    story.append(Paragraph(f"The InsOps cleaning engine applied {len(cleaning_report)} transformation(s):", body))
    story.append(Spacer(1, 0.2*cm))
    for item in cleaning_report[:20]:
        story.append(Paragraph(f"• {item}", small))

    # ── 3. Smart Insights ─────────────────────────────────────────────────────
    story += section("3. Smart Insights")
    ins_data = [["Level", "Insight"]]
    for ins in insights:
        lvl   = ins.get("level", "ok").upper()
        color = RED if lvl == "ERROR" else (ORANGE if lvl == "WARN" else GREEN)
        ins_data.append([lvl, ins["msg"]])

    ins_tbl = Table(ins_data, colWidths=[3*cm, 13*cm])
    ins_style = [
        ("BACKGROUND",   (0,0), (-1,0), DARK),
        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8),
        ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#ddd")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, LIGHT_BG]),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ]
    for i, ins in enumerate(insights, start=1):
        lvl = ins.get("level", "ok")
        c   = RED if lvl == "error" else (ORANGE if lvl == "warn" else GREEN)
        ins_style.append(("TEXTCOLOR", (0,i), (0,i), c))
        ins_style.append(("FONTNAME",  (0,i), (0,i), "Helvetica-Bold"))
    ins_tbl.setStyle(TableStyle(ins_style))
    story.append(ins_tbl)

    # ── 4. PII / PHI Detection ────────────────────────────────────────────────
    story += section("4. PII & PHI Detection Results")

    flagged = pii_df[pii_df["type"] != "Safe"]
    story.append(Paragraph(
        f"Scanned <b>{len(pii_df)}</b> columns · "
        f"<b>{len(flagged)}</b> flagged "
        f"({(pii_df['type']=='PHI').sum()} PHI, {(pii_df['type']=='PII').sum()} PII)",
        body
    ))
    story.append(Spacer(1, 0.2*cm))

    pii_tbl_data = [["Column", "Type", "Risk", "Reason", "Regulations"]]
    for _, row in pii_df.iterrows():
        pii_tbl_data.append([
            row["column"], row["type"], row["risk_level"],
            row["reason"][:55] + ("…" if len(row["reason"]) > 55 else ""),
            row["compliance_risk"] or "—"
        ])

    pii_tbl = Table(pii_tbl_data, colWidths=[3.5*cm, 1.8*cm, 1.8*cm, 7*cm, 2.9*cm])
    pii_style = [
        ("BACKGROUND",   (0,0), (-1,0), DARK),
        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 7.5),
        ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#ddd")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, LIGHT_BG]),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("WORDWRAP",     (0,0), (-1,-1), True),
    ]
    for i, (_, row) in enumerate(pii_df.iterrows(), start=1):
        c = _risk_color(row["risk_level"])
        pii_style.append(("TEXTCOLOR", (2,i), (2,i), c))
        pii_style.append(("FONTNAME",  (2,i), (2,i), "Helvetica-Bold"))
        if row["type"] != "Safe":
            pii_style.append(("TEXTCOLOR", (1,i), (1,i), RED if row["type"]=="PHI" else ORANGE))
    pii_tbl.setStyle(TableStyle(pii_style))
    story.append(pii_tbl)

    # ── 5. Compliance Scores ──────────────────────────────────────────────────
    story += section("5. Regulatory Compliance Readiness")
    regs = ["HIPAA", "GDPR", "GLBA"]
    comp_data = [["Regulation", "Score", "Status", "Notes"]]
    notes_map = {
        "HIPAA": "Governs health information (PHI)",
        "GDPR":  "EU data privacy regulation",
        "GLBA":  "US financial data privacy",
    }
    for reg in regs:
        s = compliance_scores.get(reg, 100)
        status = "COMPLIANT" if s >= 80 else ("REVIEW" if s >= 50 else "AT RISK")
        comp_data.append([reg, f"{s}/100", status, notes_map[reg]])

    comp_tbl = Table(comp_data, colWidths=[3*cm, 3*cm, 3.5*cm, 7.5*cm])
    comp_style = [
        ("BACKGROUND",   (0,0), (-1,0), DARK),
        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#ddd")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, LIGHT_BG]),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("ALIGN",        (1,0), (2,-1), "CENTER"),
    ]
    for i, reg in enumerate(regs, start=1):
        s = compliance_scores.get(reg, 100)
        c = _score_color(s)
        comp_style.append(("TEXTCOLOR", (1,i), (2,i), c))
        comp_style.append(("FONTNAME",  (1,i), (2,i), "Helvetica-Bold"))
    comp_tbl.setStyle(TableStyle(comp_style))
    story.append(comp_tbl)

    # ── 6. ML Results ─────────────────────────────────────────────────────────
    if ml_results and "metrics" in ml_results:
        story += section("6. Machine Learning Results")
        story.append(Paragraph(
            f"Model: <b>{ml_results.get('model_name','')}</b> · "
            f"Task: <b>{ml_results.get('task_type','')}</b> · "
            f"Train: {ml_results.get('train_size','')} rows · "
            f"Test: {ml_results.get('test_size','')} rows",
            body
        ))
        story.append(Spacer(1, 0.2*cm))

        ml_data = [["Metric", "Value"]]
        for k, v in ml_results["metrics"].items():
            ml_data.append([k, str(v)])

        ml_tbl = Table(ml_data, colWidths=[8*cm, 8*cm])
        ml_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0), DARK),
            ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
            ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 9),
            ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#ddd")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, LIGHT_BG]),
            ("TOPPADDING",   (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ]))
        story.append(ml_tbl)

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "This report was automatically generated by the InsOps AI Data Pipeline Platform. "
        "InsOps is the AI Small Language Model for Insurance — purpose-built for insurance data, "
        "workflows, and regulations. www.insops.com",
        ParagraphStyle("Footer", fontSize=7.5, textColor=GREY, fontName="Helvetica",
                       alignment=TA_CENTER, leading=11)
    ))

    doc.build(story)
    return buf.getvalue()

"""
app/scripts/create_sample_pdf.py
----------------------------------
Generates tests/sample.pdf — a realistic multi-section academic PDF for
use with the ingestion pipeline integration tests.

Run once:
    python -m app.scripts.create_sample_pdf
"""

from __future__ import annotations

from pathlib import Path

SAMPLE_SECTIONS = {
    "Abstract": (
        "Intermittent fasting (IF) has emerged as a promising dietary strategy "
        "for improving metabolic health. This study investigates the impact of a "
        "16:8 time-restricted eating (TRE) protocol on insulin sensitivity, fasting "
        "glucose, and HOMA-IR scores in overweight adults over a 12-week intervention."
    ),
    "Introduction": (
        "Insulin resistance is a hallmark of type 2 diabetes and metabolic syndrome. "
        "Dietary modifications that reduce postprandial glucose excursions have been "
        "shown to improve insulin signalling pathways. Intermittent fasting restricts "
        "the eating window, inducing metabolic switching from glucose to fatty-acid "
        "oxidation. Several animal models have demonstrated improved insulin sensitivity "
        "following IF protocols, yet human evidence remains heterogeneous. This study "
        "aims to fill this gap through a randomised controlled trial design."
    ),
    "Methods": (
        "One hundred adult volunteers (BMI 25-35) were randomised 1:1 to TRE "
        "(eating window 12:00-20:00) or ad libitum control for 12 weeks. Fasting "
        "venous blood samples were collected at baseline, week 6, and week 12. "
        "Insulin sensitivity was estimated using the Homeostatic Model Assessment "
        "for Insulin Resistance (HOMA-IR). Continuous glucose monitoring (CGM) "
        "devices provided 14-day glucose time-series at each visit. Statistical "
        "analysis used linear mixed-effects models with participant as random effect. "
        "The primary endpoint was change in HOMA-IR from baseline to week 12."
    ),
    "Results": (
        "The TRE group exhibited a mean HOMA-IR reduction of 1.8 units (95% CI "
        "1.2-2.4, p < 0.001) compared to 0.3 units in controls (p = 0.21). Fasting "
        "insulin decreased by 22% in TRE versus 4% in controls. CGM-derived time "
        "in range improved from 71% to 83% in TRE participants. Body weight declined "
        "by 3.2 kg in TRE versus 0.6 kg in controls, though adjustment for weight "
        "loss did not attenuate the insulin-sensitivity effect (p = 0.004). Adverse "
        "events were mild and did not differ between arms."
    ),
    "Discussion": (
        "Our findings confirm that a 16:8 TRE protocol substantially improves insulin "
        "sensitivity independent of weight loss, consistent with mechanistic studies "
        "demonstrating circadian alignment of nutrient intake with peak insulin "
        "secretion. The magnitude of HOMA-IR improvement (1.8 units) exceeds that "
        "reported for metformin monotherapy in early-stage insulin resistance. "
        "Limitations include the open-label design and reliance on self-reported "
        "eating windows. Future work should examine IF in combination with exercise "
        "and explore differential responses by chronotype."
    ),
    "Conclusion": (
        "Time-restricted eating for 12 weeks significantly reduces insulin resistance "
        "in overweight adults, with effects that persist after controlling for weight "
        "loss. These results support TRE as a scalable, low-cost metabolic intervention."
    ),
    "References": (
        "1. Patterson RE et al. (2015). Intermittent Fasting and Human Metabolic Health. "
        "JAND 115(8):1203-1212. "
        "2. Sutton EF et al. (2018). Early Time-Restricted Feeding Improves Insulin "
        "Sensitivity. Cell Metabolism 27(6):1212-1221. "
        "3. Lowe DA et al. (2020). Effects of TRE on Weight Loss. JAMA Intern Med "
        "180(11):1491-1499."
    ),
}

OUTPUT_PATH = Path("tests/sample.pdf")


def create() -> None:
    try:
        from fpdf import FPDF
    except ImportError:
        print("fpdf2 not installed. Run: pip install fpdf2")
        raise

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(
        0, 12,
        "Intermittent Fasting Improves Insulin Sensitivity: A Randomised Trial",
        ln=True, align="C",
    )
    pdf.ln(4)

    # Authors
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, "A. Author, B. Researcher, C. Scientist", ln=True, align="C")
    pdf.ln(6)

    # DOI
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, "DOI: 10.0000/test.sample", ln=True, align="C")
    pdf.ln(8)

    # Sections
    for heading, body in SAMPLE_SECTIONS.items():
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, heading, ln=True)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 7, body)
        pdf.ln(4)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT_PATH))
    print(f"Created: {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    create()

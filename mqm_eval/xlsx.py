"""Excel annotation sheets with dropdown menus.

Same columns as the CSV sheets, so the loader reads either. What the workbook adds:
  * dropdowns for no_error, dimension, subtype (depends on the chosen dimension) and severity
  * a `check` column that says whether a row is complete
  * a Guide tab with the error types and severities
Dropdowns reduce typing errors; the loader still validates every row, so a sheet edited in
another program is checked just the same.
"""
from pathlib import Path

from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.datavalidation import DataValidation

from .schema import SHEET_COLUMNS, load_typology

LAST_ROW = 800                      # dropdowns cover this many rows (room for extra error rows)
WIDTHS = {"item_id": 10, "source": 50, "translation": 50, "no_error": 10, "dimension": 24,
          "subtype": 26, "severity": 11, "span": 24, "explanation": 32, "correction": 30,
          "annotator_id": 11, "check": 34}

SUBTYPE_HELP = {
    "addition": "Content in the translation that is not in the English source.",
    "omission": "Content from the English source is missing. Write [MISSING] in span.",
    "mistranslation": "The meaning differs from the English source.",
    "untranslated": "Source text left as it is although it should be translated.",
    "grammar": "Wrong grammar in the Kumaoni.",
    "spelling": "A form you would call misspelled (not an accepted variant).",
    "punctuation": "Wrong or missing punctuation.",
    "register": "Wrong level of formality.",
    "inconsistency": "The same thing is translated differently inside the text.",
    "spelling_variant": "An accepted variant spelling. Severity must be neutral.",
    "inappropriate_for_context": "Wrong term for this context.",
    "inconsistent_use": "A term is used inconsistently.",
    "awkward": "Correct but unnatural.",
    "dialect_mismatch": "Valid Kumaoni, but not a variety you accept.",
    "hindi_leakage": "A Hindi word or form where Kumaoni has its own.",
    "other_language": "Words from another language.",
    "non_translation": "The output is unrelated to the source or unusable.",
    "source_error": "The English source itself is faulty.",
    "other": "Nothing else fits.",
}
SEVERITY_HELP = [
    ("neutral", "Differs from your preference but is acceptable. Not penalised."),
    ("minor", "A real error with limited impact; the meaning is still clear."),
    ("major", "Seriously affects understanding or meaning."),
    ("critical", "Unfit for use or risky: reverses the meaning, invents harmful facts, or is not Kumaoni."),
]
CHECK_FORMULA = ('=IF(D{r}="Y",IF(OR(E{r}<>"",F{r}<>"",G{r}<>"",H{r}<>""),"no_error=Y but errors filled","ok"),'
                 'IF(AND(E{r}="",F{r}="",G{r}="",H{r}=""),"mark errors, or put Y in no_error",'
                 'IF(OR(E{r}="",F{r}="",G{r}="",H{r}=""),"missing dimension, subtype, severity or span","ok")))')


def _put(ws, row, col, value):
    c = ws.cell(row=row, column=col)
    c.value = value
    if isinstance(value, str) and value.startswith("="):
        c.data_type = "s"            # a sentence that starts with "=" must stay text
    return c


def write_xlsx(rows, path, typology=None):
    """Write `rows` (dicts with SHEET_COLUMNS keys) as an annotation workbook."""
    typology = typology or load_typology()
    dims = typology["dimensions"]
    wb = Workbook()
    ws = wb.active
    ws.title = "Annotate"
    guide = wb.create_sheet("Guide")
    lists = wb.create_sheet("Lists")

    # --- Lists tab: dimension names in column A, one column of subtypes per dimension -------
    lists.cell(row=1, column=1, value="dimensions")
    for i, d in enumerate(dims, start=2):
        lists.cell(row=i, column=1, value=d)
    n_dim = len(dims)
    wb.defined_names["dimension_list"] = DefinedName("dimension_list", attr_text=f"Lists!$A$2:$A${n_dim + 1}")
    for j, (d, meta) in enumerate(dims.items(), start=2):
        lists.cell(row=1, column=j, value=d)
        subs = list(meta["subtypes"])
        for i, s in enumerate(subs, start=2):
            lists.cell(row=i, column=j, value=s)
        col = get_column_letter(j)
        wb.defined_names[d] = DefinedName(d, attr_text=f"Lists!${col}$2:${col}${len(subs) + 1}")
    lists.sheet_state = "hidden"

    # --- Annotate tab ------------------------------------------------------------------
    header = SHEET_COLUMNS + ["check"]
    head_fill, ro_fill = PatternFill("solid", start_color="D9E2F3"), PatternFill("solid", start_color="F2F2F2")
    for j, h in enumerate(header, start=1):
        c = ws.cell(row=1, column=j, value=h)
        c.font, c.fill = Font(bold=True), head_fill
        ws.column_dimensions[get_column_letter(j)].width = WIDTHS[h]
    ws.freeze_panes = "A2"
    ro_cols = {SHEET_COLUMNS.index(c) + 1 for c in ("item_id", "source", "translation", "annotator_id")}
    for i, r in enumerate(rows, start=2):
        for j, h in enumerate(SHEET_COLUMNS, start=1):
            c = _put(ws, i, j, r.get(h, "") or None)
            c.alignment = Alignment(wrap_text=True, vertical="top")
            if j in ro_cols:
                c.fill = ro_fill
        ws.cell(row=i, column=len(header), value=CHECK_FORMULA.format(r=i))

    def dv(formula, cols, title, msg, allow_blank=True):
        v = DataValidation(type="list", formula1=formula, allow_blank=allow_blank, showErrorMessage=True,
                           errorTitle=title, error=msg, showInputMessage=False)
        v.add(f"{cols}2:{cols}{LAST_ROW}")
        ws.add_data_validation(v)

    dv('"Y"', "D", "no_error", "Enter Y if the translation has no errors, otherwise leave blank.")
    dv("=dimension_list", "E", "dimension", "Choose a dimension from the list.")
    dv("=INDIRECT($E2)", "F", "subtype", "Choose the dimension first, then a subtype from the list.")
    dv('"neutral,minor,major,critical"', "G", "severity", "Choose neutral, minor, major or critical.")
    ws.conditional_formatting.add(
        f"L2:L{LAST_ROW}",
        FormulaRule(formula=['AND($L2<>"",$L2<>"ok")'],
                    fill=PatternFill("solid", start_color="FFC7CE", end_color="FFC7CE")))

    # --- Guide tab -----------------------------------------------------------------------
    g = guide
    g["A1"], g["A1"].font = "How to fill in the sheet", Font(bold=True, size=13)
    lines = ["One row per error. For a second error in the same item, copy the whole row and change the error columns.",
             "If the item has no errors, put Y in no_error and leave the error columns blank.",
             "Judge the Kumaoni against the English source. Do not use any translation tool.",
             "span = the exact wrong words in the translation. For an omission write [MISSING].",
             "If unsure between two severities, choose the lower and say why in explanation.",
             "The check column tells you if a row is complete. It must say ok on every row before you return the file."]
    for i, t in enumerate(lines, start=3):
        g.cell(row=i, column=1, value=f"{i - 2}. {t}")
    r0 = 3 + len(lines) + 1
    g.cell(row=r0, column=1, value="Severity").font = Font(bold=True)
    for i, (s, t) in enumerate(SEVERITY_HELP, start=r0 + 1):
        g.cell(row=i, column=1, value=s)
        g.cell(row=i, column=2, value=t)
    r1 = r0 + len(SEVERITY_HELP) + 2
    for j, h in enumerate(("dimension", "subtype", "use when"), start=1):
        g.cell(row=r1, column=j, value=h).font = Font(bold=True)
    r = r1 + 1
    for d, meta in dims.items():
        for s in meta["subtypes"]:
            g.cell(row=r, column=1, value=d)
            g.cell(row=r, column=2, value=s)
            g.cell(row=r, column=3, value=SUBTYPE_HELP.get(s, ""))
            r += 1
    g.column_dimensions["A"].width, g.column_dimensions["B"].width, g.column_dimensions["C"].width = 28, 28, 80

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path

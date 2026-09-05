"""
Funções auxiliares para consolidar a validação regional já calculada.
"""

from pathlib import Path
import re
import unicodedata

import pandas as pd


STRUCTURAL_PAIRS = [
    ("method", "C1", "C2"),
    ("method", "C3", "C4"),
    ("method", "C5", "C6"),
    ("period", "C1", "C3"),
    ("period", "C2", "C4"),
    ("source", "C3", "C5"),
    ("source", "C4", "C6")
]


EXPECTED_SHEETS = {
    "C1-C6_Todos": "structural",
    "C7-C8_Pontos": "seasonal_points",
    "C7-C8_Raster": "seasonal_raster",
    "Resumo": "summary"
}


def _normalize(value):
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def read_validation_workbook(workbook):
    """Lê todas as folhas de um Excel de validação."""

    workbook = Path(workbook)

    if not workbook.exists():
        raise FileNotFoundError(f"Excel não encontrado: {workbook}")

    return pd.read_excel(workbook, sheet_name=None)


def workbook_inventory(tables):
    """Resume as folhas e as respetivas dimensões."""

    return pd.DataFrame([
        {
            "sheet": name,
            "rows": len(table),
            "columns": len(table.columns),
            "column_names": "; ".join(map(str, table.columns)),
            "expected_scope": EXPECTED_SHEETS.get(name, "other")
        }
        for name, table in tables.items()
    ])


def sheets_to_long(tables):
    """Junta as folhas numa tabela longa sem alterar as colunas originais."""

    frames = []

    for sheet, table in tables.items():
        frame = table.copy()
        frame.insert(0, "source_sheet", sheet)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True, sort=False)


def validation_comparison_plan():
    """Devolve o plano das comparações controladas C1-C6."""

    rows = []

    for effect, scenario_a, scenario_b in STRUCTURAL_PAIRS:
        rows.append({
            "comparison_id": f"{effect}_{scenario_a}_{scenario_b}",
            "effect": effect,
            "scenario_a": scenario_a,
            "scenario_b": scenario_b,
            "interpretation_rule": (
                "Comparar os dois cenários com a mesma referência de "
                "validação e o mesmo produto."
            )
        })

    return pd.DataFrame(rows)


def _scenario_column(table):
    """Deteta a coluna que contém os identificadores C1-C8."""

    best_column = None
    best_count = 0

    for column in table.columns:
        values = table[column].astype(str).str.strip().str.upper()
        count = int(values.str.fullmatch(r"C[1-8]").sum())

        if count > best_count:
            best_column = column
            best_count = count

    return best_column, best_count


def controlled_validation_rows(tables):
    """Organiza as linhas estruturais pelos pares controlados.

    A função não assume nomes fixos de colunas. Deteta a coluna que contém
    C1-C8 e replica cada linha apenas nos pares em que o cenário participa.
    Não calcula diferenças numéricas sem conhecer a estrutura do Excel.
    """

    sheet = "C1-C6_Todos"

    if sheet not in tables:
        return pd.DataFrame(), pd.DataFrame([{
            "sheet": sheet,
            "status": "missing",
            "message": "Folha estrutural não encontrada."
        }])

    table = tables[sheet].copy()
    scenario_column, matches = _scenario_column(table)

    if scenario_column is None:
        return pd.DataFrame(), pd.DataFrame([{
            "sheet": sheet,
            "status": "scenario_column_not_found",
            "message": (
                "Não foi encontrada uma coluna com valores C1-C6. "
                "As folhas originais foram preservadas no output."
            )
        }])

    rows = []

    for _, record in table.iterrows():
        scenario = str(record[scenario_column]).strip().upper()

        for effect, scenario_a, scenario_b in STRUCTURAL_PAIRS:
            if scenario not in [scenario_a, scenario_b]:
                continue

            row = record.to_dict()
            row.update({
                "source_sheet": sheet,
                "scenario_column": str(scenario_column),
                "scenario_normalized": scenario,
                "comparison_id": f"{effect}_{scenario_a}_{scenario_b}",
                "effect": effect,
                "scenario_a": scenario_a,
                "scenario_b": scenario_b,
                "scenario_role": (
                    "A" if scenario == scenario_a else "B"
                )
            })
            rows.append(row)

    diagnostics = pd.DataFrame([{
        "sheet": sheet,
        "status": "ok",
        "scenario_column": str(scenario_column),
        "scenario_matches": matches,
        "controlled_rows": len(rows)
    }])

    return pd.DataFrame(rows), diagnostics


def seasonal_validation_rows(tables):
    """Junta as validações pontual e raster dos cenários C7-C8."""

    frames = []

    for sheet, scope in [
        ("C7-C8_Pontos", "points"),
        ("C7-C8_Raster", "raster")
    ]:
        if sheet not in tables:
            continue

        frame = tables[sheet].copy()
        frame.insert(0, "validation_scope", scope)
        frame.insert(0, "source_sheet", sheet)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)

"""
Configuração dos produtos usados nas análises C1-C8.
"""

from pathlib import Path


DATA = Path("/code/data")
RESULTS = DATA / "results"
ANALYSIS = RESULTS / "analysis"


SCENARIO_LR = {
    "C1": "C1",
    "C2": "C1",
    "C3": "C3",
    "C4": "C3",
    "C5": "C5",
    "C6": "C5"
}


def structural_maps(area):
    """Devolve os mapas estruturais disponíveis para uma região."""

    scenarios = ["C1", "C2", "C3", "C4", "C5", "C6"]

    if area == "extremadura":
        scenarios = ["C5", "C6"]

    rows = []

    for scenario in scenarios:
        if int(scenario[1:]) % 2:
            folder = RESULTS / area / scenario / "final"
            susceptibility = folder / "res_lri.tif"
            hazard = folder / "res_perigosity.tif"
            method = "LR"
        else:
            folder = RESULTS / area / scenario / "lri_model" / "class"
            susceptibility = folder / "rf_lri_base_mean_prob_1_10models.tif"
            hazard = folder / "res_perigosity_rf.tif"
            method = "LRi-RF"

        rows.extend([
            {
                "map_id": f"{scenario}_susceptibility",
                "area": area,
                "scenario": scenario,
                "product": "susceptibility",
                "method": method,
                "raster": str(susceptibility)
            },
            {
                "map_id": f"{scenario}_hazard",
                "area": area,
                "scenario": scenario,
                "product": "hazard",
                "method": method,
                "raster": str(hazard)
            }
        ])

    return rows


def seasonal_maps(area, year=2025):
    """Devolve os cinco produtos sazonais únicos de uma região."""

    c7 = RESULTS / area / "C7" / "seasonal" / "maps"
    c8 = RESULTS / area / "C8" / "seasonal" / "maps"

    return [
        {
            "map_id": f"C7_susc_{year}",
            "area": area,
            "scenario": "C7",
            "product": "seasonal_susc",
            "method": "LR",
            "raster": str(c7 / f"prob_susc_{year}.tif")
        },
        {
            "map_id": f"SSR_{year}",
            "area": area,
            "scenario": "SSR",
            "product": "ssr",
            "method": "SSR",
            "raster": str(c7 / f"prob_ssr_abs_{year}.tif")
        },
        {
            "map_id": f"C7_combined_{year}",
            "area": area,
            "scenario": "C7",
            "product": "seasonal_combined",
            "method": "LR+SSR",
            "raster": str(c7 / f"prob_combined_{year}.tif")
        },
        {
            "map_id": f"C8_susc_{year}",
            "area": area,
            "scenario": "C8",
            "product": "seasonal_susc",
            "method": "LRi-RF",
            "raster": str(c8 / f"prob_susc_{year}.tif")
        },
        {
            "map_id": f"C8_combined_{year}",
            "area": area,
            "scenario": "C8",
            "product": "seasonal_combined",
            "method": "LRi-RF+SSR",
            "raster": str(c8 / f"prob_combined_{year}.tif")
        }
    ]


def structural_pairs(pilot):
    """Devolve as comparações estruturais definidas para cada piloto."""

    if pilot == "fundao":
        base = [
            ("method", "C1", "C2"),
            ("method", "C3", "C4"),
            ("method", "C5", "C6"),
            ("period", "C1", "C3"),
            ("period", "C2", "C4"),
            ("source", "C3", "C5"),
            ("source", "C4", "C6")
        ]
    elif pilot == "badajoz":
        base = [("method", "C5", "C6")]
    else:
        raise ValueError(f"Piloto inválido: {pilot}")

    return [
        {
            "comparison_id": f"{effect}_{a}_{b}_{product}",
            "effect": effect,
            "scenario_a": a,
            "scenario_b": b,
            "product": product,
            "map_a": f"{a}_{product}",
            "map_b": f"{b}_{product}"
        }
        for effect, a, b in base
        for product in ["susceptibility", "hazard"]
    ]


def seasonal_pairs(year=2025):
    """Comparações usadas para avaliar a alteração provocada pelo SSR."""

    return [
        {
            "comparison_id": f"ssr_C7_{year}",
            "effect": "SSR",
            "scenario_a": "C7_susc",
            "scenario_b": "C7_combined",
            "product": "seasonal",
            "map_a": f"C7_susc_{year}",
            "map_b": f"C7_combined_{year}"
        },
        {
            "comparison_id": f"ssr_C8_{year}",
            "effect": "SSR",
            "scenario_a": "C8_susc",
            "scenario_b": "C8_combined",
            "product": "seasonal",
            "map_a": f"C8_susc_{year}",
            "map_b": f"C8_combined_{year}"
        }
    ]

ELEVATION_CLASSES = {
    1: "0–100 m",
    2: "100–200 m",
    3: "200–300 m",
    4: "300–400 m",
    5: "400–500 m",
    6: "500–600 m",
    7: "600–700 m",
    8: "700–800 m",
    9: "800–900 m",
    10: "900–1000 m",
    11: "1000–1500 m",
    12: "1500–2000 m"
}


SLOPE_CLASSES = {
    1: "0–5°",
    2: "5–10°",
    3: "10–15°",
    4: "15–20°",
    5: "≥20°"
}


LULC_CLASSES = {
    1: "Áreas artificiais e infraestruturas",
    2: "Culturas temporárias herbáceas e arrozais",
    3: "Vinhas",
    4: "Olivais",
    5: "Cultivos permanentes lenhosos e combinações associadas",
    6: (
        "Cultivos agrícolas genéricos, protegidos, mosaicos e "
        "agricultura com vegetação natural"
    ),
    7: (
        "Prados, pastagens, vegetação herbácea e superfícies "
        "agroflorestais"
    ),
    8: "Florestas de folhosas",
    9: "Florestas de coníferas / resinosas",
    10: "Floresta mista, outras florestas e arvoredo genérico",
    11: "Matos e combinações de vegetação natural",
    12: (
        "Vegetação esparsa, solo nu e áreas temporariamente "
        "desarborizadas"
    ),
    13: "Praias, dunas e areais",
    14: "Rocha nua, rochedos, falésias e coladas lávicas",
    15: "Água, zonas húmidas, salinas, gelo e neve",
    16: "Desconhecido / sem classificação"
}


LULC_MODELLED_CLASSES = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14
]


FEATURE_LABELS = {
    "lri_dem.tif": "Altitude",
    "lri_slope.tif": "Declive",
    "lri_lulc.tif": "Uso e ocupação do solo"
}


def model_interpretation_configurations():
    """Configurações regionais usadas na interpretação dos modelos."""

    return [
        {
            "area": "centro",
            "scenario_lr": "C1",
            "scenario_rf": "C2",
            "source": "ICNF"
        },
        {
            "area": "centro",
            "scenario_lr": "C3",
            "scenario_rf": "C4",
            "source": "ICNF"
        },
        {
            "area": "centro",
            "scenario_lr": "C5",
            "scenario_rf": "C6",
            "source": "EFFIS"
        },
        {
            "area": "extremadura",
            "scenario_lr": "C5",
            "scenario_rf": "C6",
            "source": "EFFIS"
        }
    ]


def lulc_periods(area, scenario):
    """Devolve os períodos LULC usados no cálculo LR de cada cenário."""

    base = DATA / "processed" / area

    configurations = {
        ("centro", "C1"): [
            (1995, "1995–2006", 12, "icnf", "rst_ba_1995_2006.tif"),
            (2007, "2007–2009", 3, "icnf", "rst_ba_2007_2009.tif"),
            (2010, "2010–2014", 5, "icnf", "rst_ba_2010_2014.tif"),
            (2015, "2015–2017", 3, "icnf", "rst_ba_2015_2017.tif"),
            (2018, "2018–2024", 7, "icnf", "rst_ba_2018_2024.tif")
        ],
        ("centro", "C3"): [
            (2007, "2008–2009", 2, "icnf", "rst_ba_2008_2009.tif"),
            (2010, "2010–2014", 5, "icnf", "rst_ba_2010_2014.tif"),
            (2015, "2015–2017", 3, "icnf", "rst_ba_2015_2017.tif"),
            (2018, "2018–2024", 7, "icnf", "rst_ba_2018_2024.tif")
        ],
        ("centro", "C5"): [
            (2007, "2008–2009", 2, "effis", "rst_ba_2008_2009.tif"),
            (2010, "2010–2014", 5, "effis", "rst_ba_2010_2014.tif"),
            (2015, "2015–2017", 3, "effis", "rst_ba_2015_2017.tif"),
            (2018, "2018–2024", 7, "effis", "rst_ba_2018_2024.tif")
        ],
        ("extremadura", "C5"): [
            (2005, "2008", 1, "effis", "rst_ba_2008.tif"),
            (2009, "2009–2010", 2, "effis", "rst_ba_2009_2010.tif"),
            (2011, "2011–2013", 3, "effis", "rst_ba_2011_2013.tif"),
            (2014, "2014–2016", 3, "effis", "rst_ba_2014_2016.tif"),
            (2017, "2017–2019", 3, "effis", "rst_ba_2017_2019.tif"),
            (2020, "2020–2024", 5, "effis", "rst_ba_2020_2024.tif")
        ]
    }

    key = (area, scenario)

    if key not in configurations:
        raise ValueError(f"Configuração LULC inexistente: {area} {scenario}")

    periods = []

    for year, burned_period, weight, source, burned_name in configurations[key]:
        if area == "extremadura":
            lulc_name = (
                f"lulc_siose_ar_{year}.tif"
                if year in [2017, 2020]
                else f"lulc_siose_{year}.tif"
            )
        else:
            lulc_name = f"lulc_{year}.tif"

        periods.append({
            "lulc_year": year,
            "burned_period": burned_period,
            "weight": weight,
            "lulc_raster": str(base / "lulc" / "rasters" / lulc_name),
            "burned_raster": str(
                base / "area_ardida" / source /
                "raster_count" / burned_name
            )
        })

    return periods


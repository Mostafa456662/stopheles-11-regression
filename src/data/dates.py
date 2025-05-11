bad_weather_days = [
    (
        "11",
        "12",  # Storm (Valencia)
    ),
    (
        "9",
        "25",  # Tropical Storm Hermine
    ),
    (
        "10",
        "",  # Storm
    ),
    (
        "12",
        "3",  # Extreme Cold
    ),
    (
        "12",
        "4",  # Extreme Cold
    ),
    (
        "12",
        "5",  # Extreme Cold
    ),
    (
        "7",
        "",  # Wildfires
    ),
]
bad_days = set((int(month), int(day)) for month, day in bad_weather_days if day != "")
bad_full_months = set(int(month) for month, day in bad_weather_days if day == "")

season_map = {
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Fall",
    10: "Fall",
    11: "Fall",
    12: "Winter",
}
# List of holidays (using month/day format for all years)
holidays = [
    (1, 1),  # January 1 - New Year's Day
    (1, 6),  # January 6 - Epiphany
    (4, 15),  # April 15 - Good Friday
    (5, 1),  # May 1 - Labour Day
    (8, 15),  # August 15 - Assumption of the Virgin Mary
    (10, 12),  # October 12 - National Day of Spain
    (11, 1),  # November 1 - All Saints' Day
    (12, 6),  # December 6 - Constitution Day
    (12, 8),  # December 8 - Immaculate Conception
    (12, 25),  # December 25 - Christmas Day
]

matchdays = [
    (1, 16),  # Supercopa de España Final: Athletic Bilbao 0–2 Real Madrid
    (2, 6),  # Barcelona 4–2 Atlético Madrid
    (3, 20),  # El Clásico: Real Madrid 0–4 Barcelona
    (4, 17),  # Sevilla 2–3 Real Madrid
    (5, 8),  # Madrid Derby: Atlético Madrid 1–0 Real Madrid
    (5, 7),  # Real Betis 1–2 Barcelona
    (8, 21),  # Real Sociedad 1–4 Barcelona
    (9, 18),  # Madrid Derby: Atlético Madrid 1–2 Real Madrid
    (10, 16),  # El Clásico: Real Madrid 3–1 Barcelona
    (11, 6),  # Seville Derby: Real Betis 1–1 Sevilla
    (12, 31),  # Catalan Derby: Barcelona 1–1 Espanyol
    (1, 2),  # Real Madrid 1–0 Granada
    (2, 13),  # Real Madrid 4–1 Valencia
    (3, 6),  # Atlético Madrid 2–0 Real Sociedad
    (3, 13),  # Barcelona 1–0 Osasuna
    (4, 3),  # Real Betis 1–1 Real Madrid
    (4, 10),  # Sevilla 2–2 Real Betis
    (5, 1),  # Real Sociedad 2–2 Barcelona
    (5, 15),  # Atlético Madrid 1–1 Sevilla
    (6, 5),  # Valencia 1–1 Real Madrid
    (6, 12),  # Real Betis 2–1 Barcelona
    (8, 14),  # Barcelona 4–2 Rayo Vallecano
    (8, 21),  # Real Sociedad 1–4 Barcelona
    (9, 4),  # Atlético Madrid 1–0 Getafe
    (9, 11),  # Real Madrid 2–1 Real Betis
    (9, 18),  # Atlético Madrid 1–2 Real Madrid
    (9, 25),  # Barcelona 1–0 Elche
    (10, 2),  # Real Sociedad 1–0 Atlético Madrid
    (10, 9),  # Real Madrid 2–0 Getafe
    (10, 16),  # El Clásico: Real Madrid 3–1 Barcelona
    (10, 23),  # Atlético Madrid 3–0 Real Betis
    (10, 30),  # Sevilla 1–1 Real Sociedad
    (11, 6),  # Seville Derby: Real Betis 1–1 Sevilla
    (11, 13),  # Barcelona 2–0 Girona
    (11, 20),  # Real Madrid 3–1 Cádiz
    (11, 27),  # Atlético Madrid 2–1 Real Sociedad
    (12, 4),  # Barcelona 3–0 Real Betis
    (12, 11),  # Real Madrid 2–0 Mallorca
    (12, 18),  # Atlético Madrid 1–0 Elche
    (12, 31),  # Catalan Derby: Barcelona 1–1 Espanyol
]

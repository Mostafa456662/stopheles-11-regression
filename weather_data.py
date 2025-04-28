bad_weather_days = [
    ("11", "12", # Storm (Valencia)
    ),
    ("9", "25", # Tropical Storm Hermine
    ),
    ("10", "", # Storm
    ),
    ("12", "3", # Extreme Cold
    ),
    ("12", "4", # Extreme Cold
    ),
    ("12", "5", # Extreme Cold
    ),
    ("7", "", # Wildfires
    )
]
bad_days = set((int(month), int(day)) for month, day in bad_weather_days if day != "")
bad_full_months = set(int(month) for month, day in bad_weather_days if day == "")

season_map = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 
                  7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'}

holidays = [
        (1, 1),   # January 1 - New Year's Day
        (1, 6),   # January 6 - Epiphany
        (4, 15),  # April 15 - Good Friday 
        (5, 1),   # May 1 - Labour Day
        (8, 15),  # August 15 - Assumption of the Virgin Mary
        (10, 12), # October 12 - National Day of Spain
        (11, 1),  # November 1 - All Saints' Day
        (12, 6),  # December 6 - Constitution Day
        (12, 8),  # December 8 - Immaculate Conception
        (12, 25)  # December 25 - Christmas Day
    ]
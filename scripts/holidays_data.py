import json
from pathlib import Path

REGION = "Dhanbad, Jharkhand, India"
DESCRIPTION = (
    "Localized festive and industrial holidays relevant to Dhanbad, Jharkhand. "
    "Compiled from the Jharkhand Government holiday list and regional festival calendars. "
    "Dates for lunar festivals are approximate (marked in notes). The 2017 list covers the "
    "training data year; the 2026 list covers the live forecast serving year."
)
CATEGORIES = {
    "national": "National public holiday observed across India",
    "regional": "Jharkhand state / Dhanbad regional festival with strong local observance",
    "industrial": "Industrial holiday affecting mines, factories and workshops in the Dhanbad coal belt",
}

H = [
    ("2017-01-14", "Makar Sankranti / Tusu Parab (start)", "regional", "high",
     "Tusu is a major folk festival of the Dhanbad-Jharkhand coal belt, observed over several days around Makar Sankranti (approx. Jan 14-16)."),
    ("2017-01-26", "Republic Day", "national", "medium", "Morning ceremonial load profile shift."),
    ("2017-02-24", "Maha Shivratri", "regional", "medium", "Night-long temple observance."),
    ("2017-03-12", "Holi", "regional", "high", "Holika Dahan on Mar 11 evening; Holi observed Mar 12-13 in Jharkhand."),
    ("2017-03-13", "Holi (Day 2 / Dulandi)", "regional", "high", "Second day of Holi observance."),
    ("2017-03-29", "Sarhul", "regional", "medium", "Tribal spring festival of Jharkhand (approx. date, Chaitra Shukla Tritiya)."),
    ("2017-04-04", "Ram Navami", "regional", "medium", "Large street processions across Jharkhand."),
    ("2017-04-09", "Mahavir Jayanti", "regional", "low", "Jain festival."),
    ("2017-04-14", "Good Friday / Dr. Ambedkar Jayanti", "national", "low", "Dual observance."),
    ("2017-05-01", "May Day (Labour Day)", "industrial", "medium", "Significant for the mining workforce of Dhanbad."),
    ("2017-05-10", "Buddha Purnima", "regional", "low", ""),
    ("2017-06-26", "Eid-ul-Fitr", "national", "high", "Eid prayers and feasting; observed Jun 25-26 (approx.)."),
    ("2017-06-28", "Sant Kabir Jayanti", "regional", "low", ""),
    ("2017-08-07", "Raksha Bandhan", "regional", "low", ""),
    ("2017-08-15", "Independence Day", "national", "medium", "Morning ceremonial load profile shift."),
    ("2017-08-15", "Janmashtami", "regional", "medium", "Observed Aug 14-15 in Jharkhand."),
    ("2017-09-02", "Bakrid / Eid-ul-Zuha", "national", "medium", "Observed Sep 1-2 (approx.)."),
    ("2017-09-17", "Vishwakarma Puja", "industrial", "high",
     "Fixed date (Sep 17). The biggest industrial shutdown in Dhanbad: coal mines, workshops and factories close; visible demand dip on industrial feeders."),
    ("2017-09-20", "Karma Puja", "regional", "medium", "Tribal festival on Bhadrapada Amavasya (approx. date)."),
    ("2017-09-28", "Durga Puja (Saptami)", "regional", "high",
     "Durga Puja is among the largest festivals in the Jharkhand-Bengal border belt; multi-day observance Sep 28-30."),
    ("2017-09-29", "Durga Puja (Ashtami/Navami)", "regional", "high", "Peak pandal footfall and lighting load."),
    ("2017-09-30", "Durga Puja (Vijayadashami) / Muharram", "regional", "high",
     "Dussehra immersion processions; Muharram also observed Oct 1 (approx.)."),
    ("2017-10-02", "Gandhi Jayanti", "national", "medium", ""),
    ("2017-10-17", "Dhanteras", "regional", "medium",
     "Evening lamp-lighting and shopping load spike; culturally resonant with Dhanbad's name."),
    ("2017-10-19", "Diwali (Lakshmi Puja)", "regional", "high", "Evening lighting load spike across all feeders."),
    ("2017-10-20", "Govardhan Puja / Bhai Dooj / Sohrai", "regional", "medium",
     "Sohrai, the Santal cattle festival of Jharkhand, is observed around Diwali (approx.)."),
    ("2017-10-26", "Chhath Puja (Sandhya Arghya)", "regional", "high",
     "Chhath is the single most important festival of Jharkhand; four-day observance (Nahay Khay Oct 24, Kharna Oct 25, Sandhya Arghya Oct 26, Usha Arghya Oct 27) with distinctive dawn/dusk load patterns."),
    ("2017-10-27", "Chhath Puja (Usha Arghya)", "regional", "high", "Dawn riverside observance."),
    ("2017-11-04", "Guru Nanak Jayanti", "national", "low", ""),
    ("2017-11-15", "Birsa Munda Jayanti / Jharkhand Foundation Day", "regional", "medium",
     "Statehood day of Jharkhand (Nov 15, 2000)."),
    ("2017-12-02", "Id-e-Milad", "national", "low", "Observed Dec 1-2 (approx.)."),
    ("2017-12-25", "Christmas", "national", "low", ""),
    ("2026-01-14", "Makar Sankranti / Tusu Parab (start)", "regional", "high",
     "Multi-day Tusu observance around Makar Sankranti (approx.)."),
    ("2026-01-26", "Republic Day", "national", "medium", ""),
    ("2026-02-15", "Maha Shivratri", "regional", "medium", "Approx. date."),
    ("2026-03-03", "Holi", "regional", "high", "Approx. date; Holi observed Mar 3-4, 2026."),
    ("2026-03-04", "Holi (Day 2 / Dulandi)", "regional", "high", "Approx. date."),
    ("2026-03-20", "Eid-ul-Fitr", "national", "high", "Approx. date."),
    ("2026-03-27", "Ram Navami", "regional", "medium", "Approx. date."),
    ("2026-03-27", "Sarhul", "regional", "medium", "Approx. date."),
    ("2026-04-01", "Mahavir Jayanti", "regional", "low", "Approx. date."),
    ("2026-04-03", "Good Friday", "national", "low", ""),
    ("2026-04-14", "Dr. Ambedkar Jayanti", "national", "low", ""),
    ("2026-05-01", "May Day (Labour Day)", "industrial", "medium", ""),
    ("2026-05-01", "Buddha Purnima", "regional", "low", "Approx. date."),
    ("2026-05-27", "Bakrid / Eid-ul-Zuha", "national", "medium", "Approx. date."),
    ("2026-06-26", "Muharram (Ashura)", "regional", "medium", "Approx. date."),
    ("2026-08-15", "Independence Day", "national", "medium", ""),
    ("2026-08-26", "Id-e-Milad", "national", "low", "Approx. date."),
    ("2026-08-28", "Raksha Bandhan", "regional", "low", "Approx. date."),
    ("2026-09-04", "Janmashtami", "regional", "medium", "Approx. date."),
    ("2026-09-17", "Vishwakarma Puja", "industrial", "high",
     "Fixed date (Sep 17); major industrial shutdown in the Dhanbad coal belt."),
    ("2026-09-25", "Karma Puja", "regional", "medium", "Approx. date (Bhadrapada Amavasya)."),
    ("2026-10-17", "Durga Puja (Saptami)", "regional", "high", "Approx. date; multi-day observance Oct 17-20, 2026."),
    ("2026-10-18", "Durga Puja (Ashtami)", "regional", "high", "Approx. date."),
    ("2026-10-19", "Durga Puja (Navami)", "regional", "high", "Approx. date."),
    ("2026-10-20", "Durga Puja (Vijayadashami) / Dussehra", "regional", "high", "Approx. date."),
    ("2026-10-02", "Gandhi Jayanti", "national", "medium", ""),
    ("2026-11-06", "Dhanteras", "regional", "medium", "Approx. date."),
    ("2026-11-08", "Diwali (Lakshmi Puja)", "regional", "high", "Approx. date."),
    ("2026-11-09", "Govardhan Puja / Bhai Dooj / Sohrai", "regional", "medium", "Approx. date."),
    ("2026-11-10", "Chhath Puja (Nahay Khay)", "regional", "high", "Approx. date; four-day observance Nov 10-13, 2026."),
    ("2026-11-11", "Chhath Puja (Kharna)", "regional", "high", "Approx. date."),
    ("2026-11-12", "Chhath Puja (Sandhya Arghya)", "regional", "high", "Approx. date."),
    ("2026-11-13", "Chhath Puja (Usha Arghya)", "regional", "high", "Approx. date."),
    ("2026-11-15", "Birsa Munda Jayanti / Jharkhand Foundation Day", "regional", "medium", ""),
    ("2026-11-24", "Guru Nanak Jayanti", "national", "low", "Approx. date."),
    ("2026-12-25", "Christmas", "national", "low", ""),
]


def build() -> dict:
    holidays = [
        {"date": d, "name": n, "category": c, "impact": i, "notes": notes}
        for d, n, c, i, notes in H
    ]
    return {
        "region": REGION,
        "description": DESCRIPTION,
        "categories": CATEGORIES,
        "holidays": holidays,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "holidays_dhanbad.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} with {len(H)} holiday entries")

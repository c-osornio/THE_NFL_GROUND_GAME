# THE_NFL_GROUND_GAME
Springboard Capstone 2 Project
In this project, I conducted an in-depth analysis of NFL player tracking and injury data to uncover key insights into the dynamics of the ground game. By merging multiple datasets and applying advanced statistical tests, I explored how factors such as play duration, movement patterns, and surface type (synthetic vs. natural) influence the frequency and nature of lower-extremity injuries.

Key accomplishments include:

- Performing spatial and temporal analysis to identify high-risk areas and critical play windows.
- Comparing injury patterns across different playing surfaces to inform targeted prevention strategies.
- Developing actionable recommendations to optimize training protocols and field maintenance practices.

This project not only enhanced my technical skills in Python (data wrangling, statistical analysis, data visualization) but also sharpened my ability to translate complex insights into strategic, real-world applications for improving player performance and safety.

## How to run (portfolio demo)

- Create a virtual environment (optional), then install dependencies:
  - `python -m pip install -r requirements.txt`
- Generate portfolio-ready figures (uses synthetic demo data if raw CSVs are not available):
  - `PYTHONPATH=. python scripts/generate_figures.py --demo --out-dir reports/figures`
- If you have the raw CSVs, run with real data:
  - `PYTHONPATH=. python scripts/generate_figures.py --injury-csv /path/to/InjuryRecord.csv --plays-csv /path/to/PlayList.csv --track-csv /path/to/PlayerTrackData.csv --out-dir reports/figures`

The script will save images to `reports/figures/`:
- `figure_1_injury_rate_by_surface.png`
- `figure_2_severity_distribution.png`
- `figure_3_position_irr.png`
- `figure_4_field_heatmap.png`

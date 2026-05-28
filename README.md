# Econometrics Projects 13

Partial replication project for the paper:

- Liebersohn, Jack, and Jesse Rothstein (2025), *Household mobility and mortgage rate lock*, Journal of Financial Economics.

## What is in this repository

- `submission/`: cleaned deliverables for sharing and handoff
- `analysis_outputs/`: generated tables, figures, and panel outputs
- `full_replication_homework.py`: main Python workflow that generates the full homework report
- `household_mobility_panel_analysis.py`: earlier analysis script

## Main deliverable

The main written report is:

- `submission/full_replication_answers.md`
- `submission/full_replication_answers.docx`
- `submission/full_replication_answers.tex`

## Notes

- This is a partial replication built from the public synthetic replication package, not the confidential microdata used in the published paper.
- The public repository excludes the journal PDF and the course assignment handouts.
- The public repository also excludes the large replication-package data files and the heavy generated panel CSV, so the repo stays editable and pushable on standard GitHub.
- A compiled PDF of the final report was not generated locally because no PDF engine was available in the workspace. The DOCX and TEX versions are included instead.

## Data access

The large replication package data are intentionally not tracked in this GitHub repository.

To reproduce the full project locally, download the public replication package from:

- Mendeley Data: `https://doi.org/10.17632/74sfv9kx9n.1`

After download, place the extracted folder in the project root under:

- `Data for Household Mobility and Mortgage Rate Lock/`

## Reproducibility

To regenerate the main outputs:

```bash
python3 full_replication_homework.py
```

The refreshed public deliverables are then available in `analysis_outputs/` and `submission/`.

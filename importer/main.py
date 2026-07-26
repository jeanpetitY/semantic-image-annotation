"""CLI entry point for importing semantified food data into ORKG.

Paper reference:
- The knowledge-graph construction stage that materializes food entities and
  their nutrient components before image linkage.
"""

from importer.import_data import main


if __name__ == "__main__":
    main()

try:
    from .get_from_usda import main as run_usda_enrichment
except ImportError:
    from get_from_usda import main as run_usda_enrichment


def main() -> None:
    run_usda_enrichment()


if __name__ == "__main__":
    main()

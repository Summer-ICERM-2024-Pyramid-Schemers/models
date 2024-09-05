from argparse import ArgumentParser
from time import perf_counter

from src.utils import COUNTRY_TO_LEAGUES


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--country", default="england",
                        choices=[c for c in COUNTRY_TO_LEAGUES.keys() if isinstance(c,str)],
                        help="Which country to use in the evaluation. Default is England")
    parser.add_argument("model", help="Name of model to evaluate")

    start = perf_counter()
    args = parser.parse_args()
    
    match args.model.casefold().replace("model","").replace("weighted","wtd").replace(" ","").replace("_","").replace("transfermarkt","tm").replace("regression",""):
        case "colley":
            from src.models.colleyModel import ColleyModel as model
        case "wtdcolley":
            from src.models.colleyModel import WeightedColleyModel as model
        case "massey":
            from src.models.masseyModel import MasseyModel as model
        case "wtdmassey":
            from src.models.masseyModel import WeightedMasseyModel as model
        case "bettingodds" | "betting":
            from src.models.oddsModel import BettingOddsModel as model
        case "null" | "home" | "homeadv" | "homeadvantage":
            from src.models.homeAdvModel import HomeAdvModel as model
        case "tm" | "tm1":
            from src.models.transfermarktModel import TMModelOrderedProbit as model
        case "tm2" | "tmols":
            from src.models.transfermarktModel import TMModelOrderedProbitOLSGoalDiff as model
        case _:
            print("""Model options include the following (and aliases):
    - Homeadv
    - Colley
    - Wtdcolley
    - Massey
    - Wtdmassey
    - TM
    - TMOLS
    - Bettingodds""")
            raise RuntimeError(f'No model of the name "{args.model}"!')

    model.plotBrierScores(country=args.country)
    end = perf_counter()
    print(f"Time elapsed: {end-start}")
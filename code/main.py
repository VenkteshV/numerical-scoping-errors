import argparse
from fact_checking_handler import FactCheckingHandler

def str2bool(v):
    """Custom boolean parser for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def run(args):
    fc_handler = FactCheckingHandler()
    print(f"Gold: {args.gold}")
    fc_handler.run(args.input_json_file, args.output_json_file, log=args.log, golden=args.gold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fact-checking")
    parser.add_argument("-i", "--input-json-file", dest="input_json_file", type=str, default=None, help="Path to json file to use as input")
    parser.add_argument("-o", "--output-json-file", dest="output_json_file", type=str, default=None, help="Path to json file to use for output")  
    parser.add_argument("-l", "--log", dest="log", type=str2bool, default=True, help="Whether to print logs")
    parser.add_argument("-g", "--gold", dest="gold", type=str2bool, default=False, help="Whether to use golden evidence or not")
    args = parser.parse_args()

    run(args)
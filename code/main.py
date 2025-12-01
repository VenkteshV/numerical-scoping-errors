import argparse
from fact_checking_handler import FactCheckingHandler

def run(args):
    fc_handler = FactCheckingHandler()
    fc_handler.run(args.input_json_file, args.output_json_file, log=args.log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fact-checking")
    parser.add_argument("-i", "--input-json-file", dest="input_json_file", default=None, help="Path to json file to use as input")
    parser.add_argument("-o", "--output-json-file", dest="output_json_file", default=None, help="Path to json file to use for output")
    parser.add_argument("-l", "--log", dest="log", default=True, help="Whether to print logs")
    args = parser.parse_args()

    run(args)
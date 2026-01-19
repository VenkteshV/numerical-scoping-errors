import argparse
from numerical_reasoning_handler import NumericalReasoningHandler

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
    print(f"Task: {args.task}")
    nr_handler = NumericalReasoningHandler(task=args.task, platform=args.platform)
    print(f"Gold: {args.gold}")
    nr_handler.run(args.input_json_file, args.output_json_file, log=args.log, golden=args.gold)
    print(f"Validation: {args.validation}")
    if args.validation:
        nr_handler.validate(args.input_json_file, args.output_json_file, log=args.log)
    print(f"Judge: {args.judge}")
    if args.judge:
        nr_handler.judge(args.input_json_file, args.output_json_file, log=args.log, golden=args.gold)
    if args.validation:
        nr_handler.validate_judge(args.input_json_file, args.output_json_file, log=args.log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fact-checking")
    parser.add_argument("-i", "--input-json-file", dest="input_json_file", type=str, default=None, help="Path to json file to use as input")
    parser.add_argument("-o", "--output-json-file", dest="output_json_file", type=str, default=None, help="Path to json file to use for output")  
    parser.add_argument("-l", "--log", dest="log", type=str2bool, default=True, help="Whether to print logs")
    parser.add_argument("-g", "--gold", dest="gold", type=str2bool, default=False, help="Whether to use golden evidence or not")
    parser.add_argument("-t", "--task", dest="task", type=str, default="fc", help="Task to perform (fc or qa)")
    parser.add_argument("-v", "--validation", dest="validation", type=str2bool, default=True, help="Whether to perform validation")
    parser.add_argument("-p", "--platform", dest="platform", type=str, default="local", help="Which platform to perform inference on (local or delftblue)")
    parser.add_argument("-j", "--judge", dest="judge", type=str2bool, default=False, help="Whether to perform LLM-as-a-judge on the outputs of the main inference loop")
    args = parser.parse_args()

    run(args)
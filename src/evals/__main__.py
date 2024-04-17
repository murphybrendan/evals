import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Run evaluations.")

# Add arguments
parser.add_argument('benchmark', type=str, help='Enter your name')
parser.add_argument('--model', type=str, default="claude-3-haiku", help='Enter a model name', required=False)

# Parse the arguments
args = parser.parse_args()




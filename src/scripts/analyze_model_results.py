from src.evaluation.model_analysis_utils import analyze_comparison_file
import argparse

def main():
    parser = argparse.ArgumentParser(description='Analyze model comparison results')
    parser.add_argument('file_name', type=str, 
                       help='Name of the comparison file to analyze')
    parser.add_argument('--formats', type=str, nargs='+',
                       default=['grid'],
                       help='Output formats for the table (grid, latex, github)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save tables to files')
    args = parser.parse_args()
    
    analyze_comparison_file(args.file_name, formats=args.formats)

if __name__ == "__main__":
    main() 
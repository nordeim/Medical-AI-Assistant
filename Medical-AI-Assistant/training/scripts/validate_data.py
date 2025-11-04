#!/usr/bin/env python3
"""Command-line interface for batch data validation."""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.utils.data_validator import DataValidator, MedicalDataValidator, ValidationConfig
from training.utils.validation_reporter import ValidationReporter, BatchValidationReporter


def load_data_from_file(file_path: str) -> List[Dict]:
    """Load data from various file formats."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'data' in data:
                    return data['data']
                else:
                    raise ValueError("JSON file must contain a list of records or an object with 'data' field")
        
        elif file_ext in ['.csv', '.tsv']:
            sep = ',' if file_ext == '.csv' else '\t'
            df = pd.read_csv(file_path, sep=sep, encoding='utf-8')
            return df.to_dict('records')
        
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            return df.to_dict('records')
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    except Exception as e:
        raise RuntimeError(f"Error loading {file_path}: {str(e)}")


def scan_directory_for_data_files(directory: str, extensions: List[str] = None) -> List[str]:
    """Scan directory for data files."""
    if extensions is None:
        extensions = ['.json', '.csv', '.tsv', '.xlsx', '.xls']
    
    data_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in extensions):
                data_files.append(os.path.join(root, file))
    
    return sorted(data_files)


def create_validation_config(args: argparse.Namespace) -> ValidationConfig:
    """Create validation configuration from command line arguments."""
    config = ValidationConfig()
    
    # Override defaults with command line arguments
    if args.min_text_length:
        config.min_text_length = args.min_text_length
    
    if args.max_text_length:
        config.max_text_length = args.max_text_length
    
    if args.age_min is not None and args.age_max is not None:
        config.age_range = (args.age_min, args.age_max)
    
    if args.duplicate_threshold:
        config.duplicate_similarity_threshold = args.duplicate_threshold
    
    if args.log_level:
        config.log_level = args.log_level.upper()
    
    # Custom required fields
    if args.required_fields:
        config.required_fields = args.required_fields.split(',')
    
    return config


def validate_single_file(args: argparse.Namespace) -> None:
    """Validate a single data file."""
    print(f"Validating file: {args.file}")
    
    # Load data
    try:
        data = load_data_from_file(args.file)
        print(f"Loaded {len(data)} records from {args.file}")
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    # Create validator
    config = create_validation_config(args)
    if args.medical:
        validator = MedicalDataValidator(config)
    else:
        validator = DataValidator(config)
    
    # Validate data
    result = validator.validate_dataset(data)
    
    # Generate reports
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(args.file))[0]
    html_report = os.path.join(args.output_dir, f"{base_name}_validation_report_{timestamp}.html")
    json_report = os.path.join(args.output_dir, f"{base_name}_validation_report_{timestamp}.json")
    csv_report = os.path.join(args.output_dir, f"{base_name}_validation_summary_{timestamp}.csv")
    
    # Generate reports
    reporter = ValidationReporter(config)
    
    # Data summary for reports
    data_summary = {
        'file_path': args.file,
        'record_count': len(data),
        'file_size_mb': os.path.getsize(args.file) / (1024 * 1024),
        'file_format': os.path.splitext(args.file)[1]
    }
    
    # Generate HTML report
    reporter.generate_html_report(
        result, html_report, 
        dataset_name=base_name, 
        data_summary=data_summary
    )
    print(f"HTML report generated: {html_report}")
    
    # Generate JSON report
    reporter.generate_json_report(result, json_report, data_summary)
    print(f"JSON report generated: {json_report}")
    
    # Generate CSV summary
    reporter.generate_csv_summary(result, csv_report, len(data))
    print(f"CSV summary generated: {csv_report}")
    
    # Print summary to console
    print(f"\n{'='*50}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"Status: {'PASS' if result.is_valid else 'FAIL'}")
    print(f"Score: {result.score:.1%}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    
    if result.errors:
        print(f"\nErrors:")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print(f"\nWarnings:")
        for warning in result.warnings[:5]:  # Show first 5 warnings
            print(f"  - {warning}")
        if len(result.warnings) > 5:
            print(f"  ... and {len(result.warnings) - 5} more warnings")
    
    print(f"\nReports saved to: {args.output_dir}")
    
    # Exit with appropriate code
    sys.exit(0 if result.is_valid else 1)


def validate_directory(args: argparse.Namespace) -> None:
    """Validate all data files in a directory."""
    print(f"Scanning directory: {args.directory}")
    
    # Scan for data files
    data_files = scan_directory_for_data_files(args.directory, args.extensions)
    
    if not data_files:
        print("No data files found in the specified directory.")
        sys.exit(1)
    
    print(f"Found {len(data_files)} data files")
    
    # Create validator
    config = create_validation_config(args)
    if args.medical:
        validator = MedicalDataValidator(config)
    else:
        validator = DataValidator(config)
    
    # Validate each file
    validation_results = []
    dataset_names = []
    
    for i, file_path in enumerate(data_files):
        print(f"\nValidating file {i+1}/{len(data_files)}: {os.path.basename(file_path)}")
        
        try:
            data = load_data_from_file(file_path)
            result = validator.validate_dataset(data)
            validation_results.append(result)
            dataset_names.append(os.path.splitext(os.path.basename(file_path))[0])
            
            print(f"  Score: {result.score:.1%} - {'PASS' if result.is_valid else 'FAIL'}")
            
        except Exception as e:
            print(f"  Error: {e}")
            # Create a failed result
            from training.utils.data_validator import ValidationResult
            failed_result = ValidationResult(
                is_valid=False,
                errors=[f"File processing failed: {e}"],
                score=0.0
            )
            validation_results.append(failed_result)
            dataset_names.append(os.path.splitext(os.path.basename(file_path))[0])
    
    # Generate batch reports
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate batch HTML report
    batch_reporter = BatchValidationReporter(config)
    batch_html_report = os.path.join(args.output_dir, f"batch_validation_report_{timestamp}.html")
    batch_reporter.generate_batch_summary_report(
        validation_results, batch_html_report, dataset_names
    )
    print(f"\nBatch HTML report generated: {batch_html_report}")
    
    # Generate individual reports if requested
    if args.individual_reports:
        individual_reporter = ValidationReporter(config)
        
        for i, (result, file_path) in enumerate(zip(validation_results, data_files)):
            if result.is_valid or args.include_failed:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # Data summary
                data_summary = {
                    'file_path': file_path,
                    'record_count': len(load_data_from_file(file_path)) if os.path.exists(file_path) else 0,
                    'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
                    'file_format': os.path.splitext(file_path)[1]
                }
                
                # Generate reports
                individual_html = os.path.join(args.output_dir, f"{base_name}_individual_{timestamp}.html")
                individual_json = os.path.join(args.output_dir, f"{base_name}_individual_{timestamp}.json")
                
                individual_reporter.generate_html_report(result, individual_html, base_name, data_summary)
                individual_reporter.generate_json_report(result, individual_json, data_summary)
    
    # Print batch summary
    print(f"\n{'='*50}")
    print(f"BATCH VALIDATION SUMMARY")
    print(f"{'='*50}")
    
    passed_count = sum(1 for r in validation_results if r.is_valid)
    failed_count = len(validation_results) - passed_count
    avg_score = np.mean([r.score for r in validation_results])
    
    print(f"Total files processed: {len(validation_results)}")
    print(f"Passed validation: {passed_count}")
    print(f"Failed validation: {failed_count}")
    print(f"Average score: {avg_score:.1%}")
    print(f"Success rate: {passed_count/len(validation_results):.1%}")
    
    # Show failed files
    if failed_count > 0:
        print(f"\nFailed files:")
        for i, (result, name) in enumerate(zip(validation_results, dataset_names)):
            if not result.is_valid:
                print(f"  - {name}: {result.score:.1%} ({len(result.errors)} errors)")
    
    print(f"\nBatch report saved to: {args.output_dir}")
    
    # Exit with appropriate code (0 if all passed, 1 if any failed)
    sys.exit(0 if failed_count == 0 else 1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Medical AI Training Data Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a single JSON file
  python validate_data.py --file data/training_data.json --medical --output reports/
  
  # Validate all files in a directory
  python validate_data.py --directory data/ --output reports/ --include-failed
  
  # Validate with custom settings
  python validate_data.py --file data.csv --min-text-length 20 --age-min 0 --age-max 120
        """
    )
    
    # Global arguments
    parser.add_argument('--output-dir', '-o', default='validation_reports',
                       help='Output directory for reports (default: validation_reports)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.json', '.csv', '.tsv', '.xlsx', '.xls'],
                       help='File extensions to process (default: all supported formats)')
    
    # Data validation arguments
    parser.add_argument('--min-text-length', type=int,
                       help='Minimum text length for quality checks')
    parser.add_argument('--max-text-length', type=int,
                       help='Maximum text length for quality checks')
    parser.add_argument('--age-min', type=int,
                       help='Minimum valid age')
    parser.add_argument('--age-max', type=int,
                       help='Maximum valid age')
    parser.add_argument('--duplicate-threshold', type=float,
                       help='Similarity threshold for duplicate detection (0-1)')
    parser.add_argument('--required-fields', type=str,
                       help='Comma-separated list of required fields')
    parser.add_argument('--medical', action='store_true',
                       help='Use medical-specific validation rules')
    
    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Validation commands')
    
    # File validation command
    file_parser = subparsers.add_parser('file', help='Validate a single file')
    file_parser.add_argument('file', help='Path to the data file to validate')
    file_parser.set_defaults(func=validate_single_file)
    
    # Directory validation command
    dir_parser = subparsers.add_parser('directory', help='Validate all files in a directory')
    dir_parser.add_argument('directory', help='Path to the directory containing data files')
    dir_parser.add_argument('--individual-reports', action='store_true',
                           help='Generate individual reports for each file')
    dir_parser.add_argument('--include-failed', action='store_true',
                           help='Include individual reports for failed validations')
    dir_parser.set_defaults(func=validate_directory)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()